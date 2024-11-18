// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/bias_quantization.h"

#include "core/common/common.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

Status BiasQuantization::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if ((node.OpType() != "Conv" && node.OpType() != "Gemm") || node.InputDefs().size() < 3) {
      continue;
    }

    if (!graph_utils::IsInitializer(graph, node.InputDefs()[2]->Name(), true)) {
      continue;
    }

    Node* p_dp_0 = nullptr;
    Node* p_dp_1 = nullptr;
    bool all_dqs = true;
    for (auto edge = node.InputEdgesBegin(); edge != node.InputEdgesEnd(); ++edge) {
      if (edge->GetDstArgIndex() == 0 && edge->GetNode().OpType() == QDQ::DQOpName) {
        p_dp_0 = graph.GetNode(edge->GetNode().Index());
      } else if (edge->GetDstArgIndex() == 1 && edge->GetNode().OpType() == QDQ::DQOpName) {
        p_dp_1 = graph.GetNode(edge->GetNode().Index());
      } else {
        all_dqs = false;
        break;
      }
    }

    if (!all_dqs || !p_dp_0 || !p_dp_1) {
      continue;
    }

    ONNX_NAMESPACE::TypeProto int32_type_proto;
    int32_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    ONNX_NAMESPACE::TypeProto float_type_proto;
    float_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    NodeArg& scale_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_scale"), &float_type_proto);
    graph.AddNode(graph.GenerateNodeName(node.Name() + "_scale"), "Mul", "Scale node",
                  {p_dp_0->MutableInputDefs()[1], p_dp_1->MutableInputDefs()[1]}, {&scale_node_arg}, nullptr,
                  node.Domain());
    NodeArg& bias_div_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div"), &float_type_proto);
    graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div"), "Div", "Bias div node",
                  {node.MutableInputDefs()[2], &scale_node_arg}, {&bias_div_node_arg}, nullptr, node.Domain());
    NodeArg& bias_div_round_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_div_round"), &float_type_proto);
    graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_div_round"), "Round", "Bias div round node",
                  {&bias_div_node_arg}, {&bias_div_round_node_arg}, nullptr, node.Domain());
    NodeArg& bias_int32_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_int32"), &int32_type_proto);
    Node& cast_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_int32"), "Cast", "Bias int32 node",
                                    {&bias_div_round_node_arg}, {&bias_int32_node_arg}, nullptr, node.Domain());
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
    NodeArg& bias_dq_node_arg =
        graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_bias_dq"), &float_type_proto);
    Node& dp_node = graph.AddNode(graph.GenerateNodeName(node.Name() + "_bias_dq"), QDQ::DQOpName, "Bias DQ node",
                  {&bias_int32_node_arg, &scale_node_arg}, {&bias_dq_node_arg}, nullptr, node.Domain());
    dp_node.AddAttribute("axis", static_cast<int64_t>(0));
    node.MutableInputDefs()[2] = &bias_dq_node_arg;
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
