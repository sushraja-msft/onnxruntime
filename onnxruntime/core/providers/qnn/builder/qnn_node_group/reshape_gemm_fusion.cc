// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <string>

#include "core/graph/graph_utils.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"

namespace onnxruntime {
namespace qnn {

static const NodeUnit* GetInputReshapeNodeUnit(
    const GraphViewer& graph_viewer, const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Node& gemm_node) {
  if (gemm_node.OpType() != "Gemm") {
    return nullptr;
  }
  for (auto it = gemm_node.InputEdgesBegin(); it != gemm_node.InputEdgesEnd(); it++) {
    if (it->GetDstArgIndex() == 0) {
      const Node& input_reshape_node = it->GetNode();
      if (input_reshape_node.OpType() == "Reshape" && !graph_viewer.NodeProducesGraphOutput(input_reshape_node) &&
          input_reshape_node.GetOutputEdgesCount() == 1) {
        const auto it = node_to_node_unit.find(&input_reshape_node);
        if (it != node_to_node_unit.end()) {
          const NodeUnit* input_reshape_node_unit = it->second;
          if (input_reshape_node_unit && node_unit_to_qnn_node_group.count(input_reshape_node_unit) == 0 &&
              input_reshape_node_unit->UnitType() == NodeUnit::Type::SingleNode) {
            return input_reshape_node_unit;
          }
        }
      }
    }
  }
  return nullptr;
}

static bool CheckShape(const GraphViewer& graph_viewer, const Node& input_reshape_node,
                       const Node& output_reshape_node) {
  auto get_input_shape = [](const Node& reshape_node) -> InlinedVector<int64_t> {
    auto shape = reshape_node.InputDefs()[0]->Shape();
    if (!shape) {
      return {};
    }
    InlinedVector<int64_t> input_shape;
    for (const auto& dim : shape->dim()) {
      if (dim.value_case() != ONNX_NAMESPACE::TensorShapeProto_Dimension::kDimValue) {
        return {};
      }
      input_shape.emplace_back(dim.dim_value());
    }
    return input_shape;
  };

  auto get_shape_initializer_data = [&graph_viewer](const Node& reshape_node) -> InlinedVector<int64_t> {
    const ONNX_NAMESPACE::TensorProto* shape_proto =
        graph_viewer.GetConstantInitializer(reshape_node.InputDefs()[1]->Name());
    if (!shape_proto) {
      return {};
    }
    const auto* dtype = DataTypeImpl::TensorTypeFromONNXEnum(shape_proto->data_type())->GetElementType();
    TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorProto(*shape_proto);
    Tensor tensor(dtype, shape, std::make_shared<CPUAllocator>());

    // Deserialize initializer into Tensor.
    if (onnxruntime::utils::TensorProtoToTensor(onnxruntime::Env::Default(), graph_viewer.ModelPath(), *shape_proto,
                                                tensor) != Status::OK()) {
      return {};
    }

    InlinedVector<int64_t> output;
    if (tensor.IsDataType<int64_t>()) {
      gsl::span<const int64_t> tensor_elems = tensor.DataAsSpan<int64_t>();
      output.insert(output.end(), tensor_elems.begin(), tensor_elems.end());
    } else if (tensor.IsDataType<int32_t>()) {
      gsl::span<const int32_t> tensor_elems = tensor.DataAsSpan<int32_t>();
      for (int32_t elem : tensor_elems) {
        output.emplace_back(static_cast<int64_t>(elem));
      }
    }
    return output;
  };

  InlinedVector<int64_t> input_shape = get_input_shape(input_reshape_node);
  InlinedVector<int64_t> input_shape_initializer = get_shape_initializer_data(input_reshape_node);
  InlinedVector<int64_t> output_shape_initializer = get_shape_initializer_data(output_reshape_node);
  if (input_shape.empty() || input_shape_initializer.size() != 2 ||
      input_shape.size() != output_shape_initializer.size()) {
    return false;
  }

  for (size_t i = 0; i < input_shape.size() - 1; ++i) {
    if (input_shape[i] != output_shape_initializer[i]) {
      return false;
    }
  }

  return true;
}

#define ValidateOnQnn(qnn_model_wrapper, input_reshape_node_unit, gemm_node_unit, output_reshape_node_unit, logger)   \
  CreateOrValidateOnQnn((qnn_model_wrapper), (input_reshape_node_unit), (gemm_node_unit), (output_reshape_node_unit), \
                        (logger), true)
#define CreateOnQnn(qnn_model_wrapper, input_reshape_node_unit, gemm_node_unit, output_reshape_node_unit, logger)     \
  CreateOrValidateOnQnn((qnn_model_wrapper), (input_reshape_node_unit), (gemm_node_unit), (output_reshape_node_unit), \
                        (logger), false)
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& input_reshape_node_unit,
                                    const NodeUnit& gemm_node_unit, const NodeUnit& output_reshape_node_unit,
                                    const logging::Logger& logger, bool validate) {
  assert(input_reshape_node_unit.OpType() == "Reshape" && gemm_node_unit.OpType() == "Gemm" &&
         output_reshape_node_unit.OpType() == "Reshape");
  const auto& node_name = utils::GetNodeName(gemm_node_unit);
  const NodeUnitIODef& input_def = input_reshape_node_unit.Inputs()[0];
  const NodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const NodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3;
  if (has_bias) {
    bias_def_ptr = &gemm_node_unit.Inputs()[2];
  }
  const NodeUnitIODef& output_def = output_reshape_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper weight_tensor;
  QnnTensorWrapper bias_tensor;
  QnnTensorWrapper output_tensor;
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(weight_def, weight_tensor));
  if (has_bias) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
  scalar_param.dataType = QNN_DATATYPE_BOOL_8;
  scalar_param.bool8Value = static_cast<uint8_t>(1);
  QnnParamWrapper keep_dims_param(gemm_node_unit.Index(), node_name, QNN_OP_FULLY_CONNECTED_PARAM_KEEP_DIMS,
                                  scalar_param);

  if (validate) {
    std::vector<Qnn_Tensor_t> input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(
        node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED, std::move(input_tensors),
        {output_tensor.GetQnnTensor()}, {keep_dims_param.GetQnnParam()}));
  } else {
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");
    if (has_bias) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(keep_dims_param)), "Failed to add param");
    std::vector<std::string> input_names = {input_def.node_arg.Name(), weight_def.node_arg.Name()};
    if (has_bias) {
      input_names.emplace_back(bias_def_ptr->node_arg.Name());
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                                      std::move(input_names), {output_def.node_arg.Name()},
                                                      {keep_dims_param.GetParamTensorName()}, validate),
                      "Failed to add fused Gemm node.");
  }
  return Status::OK();
}

std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const NodeUnit& gemm_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const Node& gemm_node = gemm_node_unit.GetNode();
  if (graph_viewer.NodeProducesGraphOutput(gemm_node) || gemm_node.GetOutputEdgesCount() != 1) {
    return nullptr;
  }
  const std::array<std::string_view, 1> op_types = {"Reshape"};
  const NodeUnit* output_reshape_node_unit =
      GetOnlyChildOfType(graph_viewer, gemm_node_unit, op_types, node_to_node_unit, node_unit_to_qnn_node_group);
  if (!output_reshape_node_unit) {
    return nullptr;
  }

  const NodeUnit* input_reshape_node_unit =
      GetInputReshapeNodeUnit(graph_viewer, node_to_node_unit, node_unit_to_qnn_node_group, gemm_node);
  if (!input_reshape_node_unit) {
    return nullptr;
  }

  if (!CheckShape(graph_viewer, input_reshape_node_unit->GetNode(), output_reshape_node_unit->GetNode())) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusion>(*input_reshape_node_unit, gemm_node_unit, *output_reshape_node_unit);
}

ReshapeGemmFusion::ReshapeGemmFusion(const NodeUnit& input_reshape_node_unit, const NodeUnit& gemm_node_unit,
                                     const NodeUnit& output_reshape_node_unit)
    : node_units_{} {
  node_units_[0] = &input_reshape_node_unit;
  node_units_[1] = &gemm_node_unit;
  node_units_[2] = &output_reshape_node_unit;
}

Status ReshapeGemmFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return ValidateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger);
}

Status ReshapeGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOnQnn(qmw, *node_units_[0], *node_units_[1], *node_units_[2], logger);
}

gsl::span<const NodeUnit* const> ReshapeGemmFusion::GetNodeUnits() const {
  return gsl::make_span<const NodeUnit* const>(node_units_.data(), 3);
}

const NodeUnit* ReshapeGemmFusion::GetTargetNodeUnit() const { return node_units_[1]; }

}  // namespace qnn
}  // namespace onnxruntime
