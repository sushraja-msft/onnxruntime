// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_utils.h"

#include <functional>
#include <numeric>
#include <string>
#include <vector>
#include <map>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/providers/qnn/builder/qnn_def.h"

namespace onnxruntime {
namespace qnn {
namespace utils {

size_t GetElementSizeByType(const Qnn_DataType_t& data_type) {
  const static std::unordered_map<Qnn_DataType_t, size_t> data_type_to_size = {
      {QNN_DATATYPE_INT_8, 1},
      {QNN_DATATYPE_INT_16, 2},
      {QNN_DATATYPE_INT_32, 4},
      {QNN_DATATYPE_INT_64, 8},
      {QNN_DATATYPE_UINT_8, 1},
      {QNN_DATATYPE_UINT_16, 2},
      {QNN_DATATYPE_UINT_32, 4},
      {QNN_DATATYPE_UINT_64, 8},
      {QNN_DATATYPE_FLOAT_16, 2},
      {QNN_DATATYPE_FLOAT_32, 4},
      {QNN_DATATYPE_BOOL_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_16, 2},
      {QNN_DATATYPE_SFIXED_POINT_32, 4},
      {QNN_DATATYPE_UFIXED_POINT_8, 1},
      {QNN_DATATYPE_UFIXED_POINT_16, 2},
      {QNN_DATATYPE_UFIXED_POINT_32, 4},
  };

  auto pos = data_type_to_size.find(data_type);
  ORT_ENFORCE(pos != data_type_to_size.end(), "Unknown QNN data type", data_type);
  return pos->second;
}
size_t GetElementSizeByType(ONNXTensorElementDataType elem_type) {
  const static std::unordered_map<ONNXTensorElementDataType, size_t> elem_type_to_size = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, sizeof(Int4x2)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, sizeof(UInt4x2)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, sizeof(int8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, sizeof(int16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, sizeof(int32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, sizeof(int64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, sizeof(uint8_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, sizeof(uint16_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, sizeof(uint32_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, sizeof(uint64_t)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, 2},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, sizeof(float)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, sizeof(double)},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, sizeof(bool)}};

  auto pos = elem_type_to_size.find(elem_type);
  ORT_ENFORCE(pos != elem_type_to_size.end(), "Unknown element type", elem_type);
  return pos->second;
}

Status GetQnnDataType(const bool is_quantized_tensor, const ONNX_NAMESPACE::TypeProto* type_proto,
                      Qnn_DataType_t& tensor_data_type) {
  if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "The tensor doesn't have elem_type.");
  }

  int32_t onnx_data_type = type_proto->tensor_type().elem_type();
  ORT_RETURN_IF_NOT(OnnxDataTypeToQnnDataType(onnx_data_type, tensor_data_type, is_quantized_tensor),
                    "Failed to map Onnx data type to Qnn data type!");

  return Status::OK();
}

const std::string& GetNodeName(const NodeUnit& node_unit) {
  const std::string& node_name = node_unit.Name();
  if (node_name.empty()) {
    return node_unit.Outputs()[0].node_arg.Name();
  }

  return node_name;
}

bool OnnxDataTypeToQnnDataType(const int32_t onnx_data_type, Qnn_DataType_t& qnn_data_type, bool is_quantized) {
  const std::unordered_map<int32_t, Qnn_DataType_t> onnx_to_qnn_data_type = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, QNN_DATATYPE_INT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, QNN_DATATYPE_INT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, QNN_DATATYPE_INT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, QNN_DATATYPE_INT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, QNN_DATATYPE_UINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, QNN_DATATYPE_UINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, QNN_DATATYPE_UINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, QNN_DATATYPE_UINT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, QNN_DATATYPE_FLOAT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, QNN_DATATYPE_FLOAT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, QNN_DATATYPE_BOOL_8},
  };

  const std::unordered_map<int32_t, Qnn_DataType_t> onnx_to_qnn_data_type_quantized = {
      {ONNX_NAMESPACE::TensorProto_DataType_INT4, QNN_DATATYPE_SFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT8, QNN_DATATYPE_SFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_INT16, QNN_DATATYPE_SFIXED_POINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_INT32, QNN_DATATYPE_SFIXED_POINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_INT64, QNN_DATATYPE_INT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT4, QNN_DATATYPE_UFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT8, QNN_DATATYPE_UFIXED_POINT_8},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT16, QNN_DATATYPE_UFIXED_POINT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT32, QNN_DATATYPE_UFIXED_POINT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_UINT64, QNN_DATATYPE_UINT_64},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, QNN_DATATYPE_FLOAT_16},
      {ONNX_NAMESPACE::TensorProto_DataType_FLOAT, QNN_DATATYPE_FLOAT_32},
      {ONNX_NAMESPACE::TensorProto_DataType_BOOL, QNN_DATATYPE_BOOL_8},
  };

  const auto do_type_mapping = [](const std::unordered_map<int32_t, Qnn_DataType_t>& mapping_table,
                                  const int32_t onnx_data_type,
                                  Qnn_DataType_t& qnn_data_type) -> bool {
    auto pos = mapping_table.find(onnx_data_type);
    if (pos == mapping_table.end()) {
      return false;
    }
    qnn_data_type = pos->second;
    return true;
  };

  if (is_quantized) {
    return do_type_mapping(onnx_to_qnn_data_type_quantized, onnx_data_type, qnn_data_type);
  } else {
    return do_type_mapping(onnx_to_qnn_data_type, onnx_data_type, qnn_data_type);
  }
}

std::pair<float, float> CheckMinMax(float rmin, float rmax) {
  // Ensure a minimum range of 0.0001 (required by QNN)
  rmax = std::max(rmax, rmin + 0.0001f);

  // Both QNN and ORT require the range to include 0.0f
  rmin = std::min(rmin, 0.0f);
  rmax = std::max(rmax, 0.0f);

  return std::make_pair(rmin, rmax);
}

template <typename T>
Status GetQminQmax(const Qnn_DataType_t qnn_data_type,
                   T& qmin,
                   T& qmax) {
  if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_8) {
    qmin = static_cast<T>(std::numeric_limits<int8_t>::min());
    qmax = static_cast<T>(std::numeric_limits<int8_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_UFIXED_POINT_8) {
    qmin = static_cast<T>(std::numeric_limits<uint8_t>::min());
    qmax = static_cast<T>(std::numeric_limits<uint8_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_16) {
    qmin = static_cast<T>(std::numeric_limits<int16_t>::min());
    qmax = static_cast<T>(std::numeric_limits<int16_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
    qmin = static_cast<T>(std::numeric_limits<uint16_t>::min());
    qmax = static_cast<T>(std::numeric_limits<uint16_t>::max());
  } else if (qnn_data_type == QNN_DATATYPE_SFIXED_POINT_32) {
    qmin = static_cast<T>(std::numeric_limits<int32_t>::min());
    qmax = static_cast<T>(std::numeric_limits<int32_t>::max());
  } else {
    ORT_RETURN_IF(true, "Qnn Data Type: %d not supported yet.", qnn_data_type);
  }
  return Status::OK();
}

Status GetQuantParams(float rmin,
                      float rmax,
                      const Qnn_DataType_t qnn_data_type,
                      float& scale,
                      int32_t& zero_point,
                      bool symmetric) {
  std::tie(rmin, rmax) = CheckMinMax(rmin, rmax);
  if (symmetric) {
    float abs_max = std::max(abs(rmax), abs(rmin));
    rmax = abs_max;
    rmin = -abs_max;
  }

  float qmin = 0.0f;
  float qmax = 255.0f;
  ORT_RETURN_IF_ERROR(GetQminQmax(qnn_data_type, qmin, qmax));

  scale = (rmax - rmin) / (qmax - qmin);
  float initial_zero_point = 0.0f;
  if (symmetric) {
    initial_zero_point = std::round(rmin + rmax) / 2;
  } else {
    initial_zero_point = qmin - (rmin / scale);
  }
  zero_point = static_cast<int32_t>(RoundHalfToEven(Saturate(qmax, qmin, initial_zero_point)));
  // To match QNN quantization definition
  zero_point = 0 - zero_point;
  return Status::OK();
}

double Dequantize(int32_t offset, float scale, const double quant_value) {
  double offset_d = static_cast<double>(offset);
  double scale_d = static_cast<double>(scale);
  return (quant_value + offset_d) * scale_d;
}

Status Quantize(const double double_value,
                const float scale,
                const int32_t zero_point,
                const Qnn_DataType_t qnn_data_type,
                int& quant_value) {
  int qmin = 0;
  int qmax = 255;
  ORT_RETURN_IF_ERROR(GetQminQmax(qnn_data_type, qmin, qmax));
  quant_value = Saturate(qmax, qmin, static_cast<int>(std::round((double_value / scale) - zero_point)));
  return Status::OK();
}

// NCHW shape to channel last
Status NchwShapeToNhwc(const std::vector<uint32_t>& nchw_shape, std::vector<uint32_t>& nhwc_shape) {
  ORT_RETURN_IF_NOT(nchw_shape.size() == 4, "shape should have 4 dimension NCHW.");
  nhwc_shape[0] = nchw_shape[0];
  nhwc_shape[1] = nchw_shape[2];
  nhwc_shape[2] = nchw_shape[3];
  nhwc_shape[3] = nchw_shape[1];

  return Status::OK();
}

// NCHW shape to HWCN shape, required for Conv weight
Status NchwShapeToHwcn(const std::vector<uint32_t>& nchw_shape, std::vector<uint32_t>& hwcn_shape) {
  if (nchw_shape.size() == 4) {
    hwcn_shape[0] = nchw_shape[2];
    hwcn_shape[1] = nchw_shape[3];
    hwcn_shape[2] = nchw_shape[1];
    hwcn_shape[3] = nchw_shape[0];
  } else if (nchw_shape.size() == 5) {
    hwcn_shape[0] = nchw_shape[2];
    hwcn_shape[1] = nchw_shape[3];
    hwcn_shape[2] = nchw_shape[4];
    hwcn_shape[3] = nchw_shape[1];
    hwcn_shape[4] = nchw_shape[0];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported rank! only support 4 or 5.");
  }

  return Status::OK();
}

// CNHW shape to HWCN shape, required for Conv weight
Status CnhwShapeToHwcn(const std::vector<uint32_t>& cnhw_shape, std::vector<uint32_t>& hwcn_shape) {
  if (cnhw_shape.size() == 4) {
    hwcn_shape[0] = cnhw_shape[2];
    hwcn_shape[1] = cnhw_shape[3];
    hwcn_shape[2] = cnhw_shape[0];
    hwcn_shape[3] = cnhw_shape[1];
  } else if (cnhw_shape.size() == 5) {
    hwcn_shape[0] = cnhw_shape[2];
    hwcn_shape[1] = cnhw_shape[3];
    hwcn_shape[2] = cnhw_shape[4];
    hwcn_shape[3] = cnhw_shape[0];
    hwcn_shape[4] = cnhw_shape[1];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported rank! only support 4 or 5.");
  }

  return Status::OK();
}

namespace {
Status TransposeInitializer(const QnnModelWrapper& qnn_model_wrapper, const onnx::TensorProto& initializer,
                            const std::vector<size_t>& perm, std::vector<uint8_t>& transposed_data) {
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(initializer.data_type())->GetElementType();
  const auto tensor_shape_dims = onnxruntime::utils::GetTensorShapeFromTensorProto(initializer);
  TensorShape tensor_shape{tensor_shape_dims};
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  Tensor in_tensor = Tensor(tensor_dtype, tensor_shape, cpu_allocator);

  auto rank = perm.size();
  std::vector<int64_t> new_tensor_shape_dims;
  std::vector<size_t> permutations;
  new_tensor_shape_dims.reserve(rank);
  permutations.reserve(rank);
  for (int64_t p : perm) {
    permutations.push_back(p);
    new_tensor_shape_dims.push_back(tensor_shape_dims[p]);
  }

  TensorShape new_tensor_shape(new_tensor_shape_dims);
  Tensor out_tensor = Tensor(tensor_dtype, new_tensor_shape, cpu_allocator);
  ORT_RETURN_IF_ERROR(onnxruntime::utils::TensorProtoToTensor(
      Env::Default(), qnn_model_wrapper.GetGraphViewer().ModelPath(), initializer, in_tensor));
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(permutations, in_tensor, out_tensor));
  onnx::TensorProto new_tensor_proto = onnxruntime::utils::TensorToTensorProto(out_tensor, "test");
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(new_tensor_proto, transposed_data));

  return Status::OK();
}
}  // namespace

Status TransposeFromNchwToHwcn(const QnnModelWrapper& qnn_model_wrapper, const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data, bool is_3d) {
  auto& perm = is_3d ? nchw2hwcn_perm_3d : nchw2hwcn_perm;
  return TransposeInitializer(qnn_model_wrapper, initializer, perm, transposed_data);
}

Status TransposeFromCnhwToHwcn(const QnnModelWrapper& qnn_model_wrapper, const onnx::TensorProto& initializer,
                               std::vector<uint8_t>& transposed_data, bool is_3d) {
  auto& perm = is_3d ? cnhw2hwcn_perm_3d : cnhw2hwcn_perm;
  return TransposeInitializer(qnn_model_wrapper, initializer, perm, transposed_data);
}

Status TwoDimensionTranspose(const QnnModelWrapper& qnn_model_wrapper, std::vector<uint32_t>& data_shape,
                             const onnx::TensorProto& initializer, std::vector<uint8_t>& transposed_data) {
  auto tmp = data_shape[0];
  data_shape[0] = data_shape[1];
  data_shape[1] = tmp;
  std::vector<size_t> two_dim_trans_perm{1, 0};
  return TransposeInitializer(qnn_model_wrapper, initializer, two_dim_trans_perm, transposed_data);
}

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
