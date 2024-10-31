// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(ExecutionProvider);
ORT_RUNTIME_CLASS(ExecutionProviderFactory);
ORT_RUNTIME_CLASS(Node);
ORT_RUNTIME_CLASS(Graph);
ORT_RUNTIME_CLASS(GraphViewer);

typedef struct OrtCreateStream {
  int device_type;
  void*(ORT_API_CALL* CreateStreamFunc)(const OrtDevice*);
} OrtCreateStream;

typedef struct OrtMetaDef {
  char* name;
  char* domain;
  int since_version;

  char** inputs;
  size_t input_len;
  char** outputs;
  size_t output_len;
  char** constant_initializers;
  size_t initializer_len;

  char* doc_string;
} OrtMetaDef;

typedef struct OrtIndexedSubGraph {
  OrtMetaDef* meta_def; // TODO(leca): how to define a nested structure pointer?
  size_t* node_index;
  size_t node_index_len;
} OrtIndexedSubGraph;

typedef struct OrtComputeContext {
  void*(ORT_API_CALL* AllocateFunc)(void*, size_t, size_t);
  void(ORT_API_CALL* DestroyFunc)(void*, void*);
  void* allocator_handle;
  const char* node_name;
} OrtComputeContext;

typedef struct OrtNodeComputeInfo {
  int(ORT_API_CALL* CreateFunctionStateFunc)(OrtComputeContext*, void*, void**);
  OrtStatusPtr(ORT_API_CALL* ComputeFunc)(void*, void*, const OrtApi*, OrtKernelContext*);
  void(ORT_API_CALL* DestroyFunctionStateFunc)(void*);
} OrtNodeComputeInfo;

typedef struct OrtTensorRef {   // TODO(leca): OrtValueInfoRef inside OrtTensorRef?
  int64_t* shape;
  size_t shape_len;
  ONNXTensorElementDataType data_type;
  const char* data;
  size_t data_len;
} OrtTensorRef;

typedef struct OrtValueInfoRef {
  int64_t* shape;
  size_t shape_len;
  ONNXTensorElementDataType data_type;
} OrtValueInfoRef;

typedef struct OrtExecutionProvider {
#ifdef __cplusplus
  OrtExecutionProvider() : GetCapability{nullptr}, Compile{nullptr}, RegisterKernels{nullptr}, CanCopy{nullptr}, CopyTensor{nullptr}, CreatePreferredAllocators{nullptr}, type{nullptr}, create_stream{nullptr}, default_device{nullptr},
                           extra_param_for_create_state_func{nullptr}, extra_param_for_compute_func{nullptr} {}
#endif
  void(ORT_API_CALL* GetCapability)(const OrtExecutionProvider* this_, const OrtGraphViewer* graph, size_t* cnt, OrtIndexedSubGraph***);
  OrtStatusPtr(ORT_API_CALL* Compile)(OrtExecutionProvider* this_, const OrtGraphViewer** graph, const OrtNode** node, size_t cnt, OrtNodeComputeInfo* node_compute_info);
  void(ORT_API_CALL* RegisterKernels)(OrtKernelRegistry* kernel_registry);
  bool(ORT_API_CALL* CanCopy)(const OrtDevice* source, const OrtDevice* target);
  OrtStatusPtr(ORT_API_CALL* CopyTensor)(const void* src, OrtMemoryInfoDeviceType source_device_type, OrtMemoryType source_mem_type, void* dst, OrtMemoryInfoDeviceType target_device_type, size_t count, void* stream);
  int(ORT_API_CALL* CreatePreferredAllocators)(OrtExecutionProvider* this_, OrtAllocator*** ort_allocators);
  void(ORT_API_CALL* ReleaseIndexedSubGraphs)(OrtIndexedSubGraph** indexed_sub_graphs, size_t num_sub_graph);
  const char* type;
  OrtCreateStream* create_stream;
  const OrtDevice* default_device;
  void* extra_param_for_create_state_func;
  void* extra_param_for_compute_func;
} OrtExecutionProvider;

typedef struct OrtExecutionProviderFactory {
  OrtExecutionProvider*(ORT_API_CALL* CreateExecutionProvider)(OrtExecutionProviderFactory* this_, const char* const* ep_option_keys, const char* const* ep_option_values, size_t option_size);
} OrtExecutionProviderFactory;

struct OrtGraphApi {
/** \brief Get the graph name
 *
 * \param[in] graph The graph to query
 * \param[out] out The name of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_GetName, const OrtGraphViewer* graph, _Outptr_ const char** out);

/** \brief Check if the name is a constant initializer of the graph
 *
 * \param[in] graph The graph to query
 * \param[in] name The name to check
 * \param[in] check_outer_scope If true and 'graph' is a subgraph, check ancestor graph/s for 'name' if not found in 'graph'.
 * \param[out] out True if the name is a constant initializer of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_IsConstantInitializer, const OrtGraphViewer* graph, const char* name, bool check_outer_scope, _Out_ bool* out);

/** \brief Get the NodeIndex values of the graph nodes sorted in topological order
 *
 * \param[in] graph The graph to query
 * \param[in] execution_order The execution order can be 0, 1 or 2
 *                           0 means the nodes are sorted in topological order.
 *                           1 means the nodes are sorted in topological order with priority.
 *                           2 means the nodes are sorted in memory efficient topological order.
 * \param[out] nodes_index_in_topological_order The NodeIndex values of the graph nodes sorted in topological order
 * \param[out] num_nodes The number of nodes
 *
 */
ORT_API2_STATUS(OrtGraph_GetNodesIndexInTopologicalOrder, const OrtGraphViewer* graph, int execution_order, _Out_ const size_t** nodes_index_in_topological_order, _Out_ size_t* num_nodes);

/** \brief Check if the graph is a subgraph
 *
 * \param[in] graph The graph to query
 * \param[out] out True if the graph is a subgraph
 *
 */
ORT_API2_STATUS(OrtGraph_IsSubgraph, const OrtGraph* graph, _Out_ bool* out);

/** \brief Get the parent graph of the graph
 *
 * \param[in] graph The graph to query
 * \param[out] parent_graph The parent graph of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_GetParentGraph, const OrtGraph* graph, _Outptr_ const OrtGraph** parent_graph);

/** \brief Check if the graph is a subgraph
 * TODO(leca): maybe deprecate OrtGraph_IsSubgraph?
 *
 * \param[in] graph The graph to query
 * \param[out] out True if the graph is a subgraph
 *
 */
ORT_API2_STATUS(OrtGraph_IsSubgraph2, const OrtGraphViewer* graph, _Out_ bool* out);

/** \brief Get the parent node of the graph
 *
 * \param[in] graph The graph to query
 * \param[out] parent_node The node containing this Graph if IsSubgraph is true. Returns nullptr otherwise.
 *
 */
ORT_API2_STATUS(OrtGraph_GetParenNode, const OrtGraphViewer* graph, _Outptr_ const OrtNode** parent_node);

/** \brief Gets the path of the owning model if any
 *
 * \param[in] graph The graph to query
 * \param[out] model_path The path of the owning model if any
 *
 */
ORT_API2_STATUS(OrtGraph_GetModelPath, const OrtGraphViewer* graph, _Outptr_ const void** model_path);

/** \brief Get the internal graph in the graph viewer
 *
 * \param[in] graph_viewer The graph viewer to query
 * \param[out] graph The internal graph in the graph viewer
 *
 */
ORT_API2_STATUS(OrtGraph_GetOrtGraph, const OrtGraphViewer* graph_viewer, _Outptr_ const OrtGraph** graph);

/** \brief Gets the Graph inputs with no matching initializers, in the same order as defined in the GraphProto.
 *
 *  NOTE!!: The caller is responsible for releasing the char array using ReleaseCharArray.
 *
 * \param[in] graph The graph to query
 * \param[out] input_names The input names
 * \param[out] input_len The number of inputs
 *
 */
ORT_API2_STATUS(OrtGraph_GetRequiredInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len);

/** \brief Gets the Graph inputs with matching initializers, in the same order as defined in the GraphProto.
 *
 *  NOTE!!: The caller is responsible for releasing the char array using ReleaseCharArray.
 *
 * \param[in] graph The graph to query
 * \param[out] input_names The input names
 * \param[out] input_len The number of inputs
 *
 */
ORT_API2_STATUS(OrtGraph_GetAllInputs, const OrtGraphViewer* graph, _Outptr_ const char*** input_names, _Out_ size_t* input_len);

/** \brief Gets all the Graph initializers' name
 *
 *  NOTE!!: The caller is responsible for releasing the char array using ReleaseCharArray.
 *
 * \param[in] graph The graph to query
 * \param[out] initializer_names The initializer names
 * \param[out] initializer_len The number of initializers
 *
 */
ORT_API2_STATUS(OrtGraph_GetAllInitializers, const OrtGraphViewer* graph, _Outptr_ const char*** initializer_names, _Out_ size_t* initializer_len);

// TODO(leca): maybe OrtGraph_ReleaseCharArray?
/** \brief Release the char array
 *
 *  NOTE!!: Invoke this function after the use of OrtGraph_GetRequiredInputs, OrtGraph_GetAllInputs, OrtGraph_GetAllInitializers.
 *
 * \param[in] char_array The char array to release
 *
 */
ORT_API2_STATUS(ReleaseCharArray, const char** char_array);

/** \brief Get const Node given specific node index. May return nullptr if node as been freed.
 *
 * \param[in] graph The graph to query
 * \param[in] node_index The index of the node
 * \param[out] node The node
 *
 */
ORT_API2_STATUS(OrtGraph_GetOrtNode, const OrtGraphViewer* graph, size_t node_index, _Outptr_ const OrtNode** node);

/** \brief Get the consumer nodes of a node arg with the given name
 *
 * \param[in] graph The graph to query
 * \param[in] input_name The name of the node arg
 * \param[out] consumers The consumer nodes of the node arg
 * \param[out] num_consumers The number of consumer nodes
 *
 */
ORT_API2_STATUS(OrtGraph_GetNodesConsumingInput, const OrtGraphViewer* graph, const char* input_name, _Outptr_ const OrtNode*** consumers, _Out_ size_t* num_consumers); // TODO(leca): ValueConsumers::comprehensive ?

/** \brief Get the producer node of a node arg with the given name
 *
 * \param[in] graph The graph to query
 * \param[in] output_name The name of the node arg
 * \param[out] node The node producing the node arg
 *
 */
ORT_API2_STATUS(OrtGraph_GetNodeProducingOutput, const OrtGraphViewer* graph, const char* output_name, _Outptr_ const OrtNode** node);

/** \brief Gets the number of valid Nodes in the Graph.
 *
 * \param[in] graph The graph to query
 * \param[out] num_nodes The number of valid nodes in the graph
 *
 */
ORT_API2_STATUS(OrtGraph_NumberOfNodes, const OrtGraphViewer* graph, _Out_ int* num_nodes);

/** \brief Gets the maximum NodeIndex value used in the Graph.
 *
 * \param[in] graph The graph to query
 * \param[out] max_node_index The maximum NodeIndex value used by Nodes in the Graph
 *
 */
ORT_API2_STATUS(OrtGraph_MaxNodeIndex, const OrtGraphViewer* graph, _Out_ int* max_node_index);

/** \brief Gets the number of outputs of the Graph.
 *
 * \param[in] graph The graph to query
 * \param[out] output_len The number of outputs of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_GetOutputSize, const OrtGraphViewer* graph, _Out_ size_t* output_len);

/** \brief Gets the name of the i-th output of the Graph.
 *
 * \param[in] graph The graph to query
 * \param[in] i The index of the output
 * \param[out] out The name of the i-th output of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_GetIthOutputName, const OrtGraphViewer* graph, size_t i, _Outptr_ const char** out);

/** \brief Gets the element type of the i-th output of the Graph.
 *
 * \param[in] graph The graph to query
 * \param[in] i The index of the output
 * \param[out] out The element type of the i-th output of the graph
 *
 */
ORT_API2_STATUS(OrtGraph_GetIthOutputElemType, const OrtGraphViewer*, size_t i, _Out_ int32_t* out);

/** \brief Gets the initializer tensor of the Graph.
 *
 *  NOTE!!: The caller is responsible for releasing the initializer tensor using OrtGraph_ReleaseInitializerTensor.
 *
 * \param[in] graph The graph to query
 * \param[in] initializer_name The name of the initializer tensor
 * \param[out] out The initializer tensor
 *
 */
ORT_API2_STATUS(OrtGraph_GetInitializerTensor, const OrtGraphViewer* graph, const char* initializer_name, _Outptr_ OrtTensorRef**);

/** \brief Release the initializer tensor.
 *
 *  NOTE!!: Invoke this function after the use of OrtGraph_GetInitializerTensor.
 *
 * \param[in] tensor The initializer tensor to release
 *
 */
ORT_API2_STATUS(OrtGraph_ReleaseInitializerTensor, OrtTensorRef* tensor);

// TODO(leca): Do we need to define and expose OrtValueInfoRef?
// We can also encapsulate it, provide input/output index or name, return the properties of OrtValueInfoRef(shape, data_type)
// Just like OrtGraph_GetIthOutputElemType
/** \brief Gets the value info of the node arg with the given name.
 *
 * NOTE!!: The caller is responsible for releasing the value info using OrtGraph_ReleaseValueInfo.
 *
 * \param[in] graph The graph to query
 * \param[in] name The name of the node arg
 * \param[out] out The value info
 *
 */
ORT_API2_STATUS(OrtGraph_GetValueInfo, const OrtGraphViewer* graph, const char* name, _Outptr_ OrtValueInfoRef** out);

/** \brief Release the value info.
 *
 *  NOTE!!: Invoke this function after the use of OrtGraph_GetValueInfo.
 *
 * \param[in] value_info The value info to release
 *
 */
ORT_API2_STATUS(OrtGraph_ReleaseValueInfo, OrtValueInfoRef* value_info);

/** \brief Serialize the Graph to a byte array.
 *
 * \param[in] graph The graph to serialize
 * \param[out] data The byte array
 * \param[out] data_size The size of the byte array
 *
 * \remarks The caller is responsible for freeing the byte array using OrtFreeMem.
 *
 */
ORT_API2_STATUS(OrtGraph_SerializeToArray, const OrtGraphViewer* graph, _Out_ void** data, _Out_ size_t* data_size);  // TODO(leca): review and discuss

/** \brief Construct a subgraph from the Graph with the given node indices.
 *
 * \param[in] graph The graph to query
 * \param[in] node_num The number of node indices
 * \param[in] node_indices The indices of the nodes to include in the subgraph
 * \param[out] subgraph The constructed subgraph
 *
 * \remarks The caller is responsible for releasing the subgraph using OrtGraph_ReleaseGraph.
 *
 */
ORT_API2_STATUS(OrtGraph_GetSubGraph, const OrtGraphViewer* graph, const int node_num, const size_t* node_indices, _Outptr_ const OrtGraphViewer** subgraph); // TODO(yang): review and discuss

/** \brief Release the graph.
 *
 * NOTE!!: Invoke this function after the use of OrtGraph_GetSubGraph. As OrtGraph_GetSubGraph allocate model instead of
 * graph, this API release graph's owning_model explicitly which in turn will release the graph
 * (because graph is hosted in an unique_ptr in Model class)
 *
 * \param[in] graph The graph to release
 *
 */
ORT_API2_STATUS(OrtGraph_ReleaseGraph, const OrtGraphViewer* graph);

/** \brief Gets the name of the node
 *
 * \param[in] node The node to query
 * \param[out] out The name of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetName, const OrtNode* node, _Outptr_ const char** out);

/** \brief Gets the description of the node
 *
 * \param[in] node The node to query
 * \param[out] out The description of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetDescription, const OrtNode* node, _Outptr_ const char** out);

/** \brief Gets the domain of the node
 *
 * \param[in] node The node to query
 * \param[out] out The domain of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetDomain, const OrtNode* node, _Outptr_ const char** out);

/** \brief Gets the opset version that the Node's operator was first defined in.
 *
 * \param[in] node The node to query
 * \param[out] out The since version of the node
 *
 */
ORT_API2_STATUS(OrtNode_SinceVersion, const OrtNode* node, _Out_ int* out);

/** \brief Gets the execution ProviderType that this node will be executed by.
 *
 * \param[in] node The node to query
 * \param[out] out The execution ProviderType of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetExecutionProviderType, const OrtNode* node, _Out_ const char** out);

/** \brief Gets the Node's operator type.
 *
 * \param[in] node The node to query
 * \param[out] out The operator type of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetOpType, const OrtNode* node, _Outptr_ const char** out);

/** \brief Gets the number of implicit inputs of the node.
 *
 * \param[in] node The node to query
 * \param[out] out The number of implicit inputs of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetImplicitInputSize, const OrtNode* node, _Out_ size_t* out);

/** \brief Gets the i-th implicit input name of the node.
 *
 * \param[in] node The node to query
 * \param[in] i The index of the implicit input
 * \param[out] out The i-th implicit input name of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetIthImplicitInputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

/** \brief Gets the number of inputs of the node.
 *
 * \param[in] node The node to query
 * \param[out] out The number of inputs of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetNumInputs, const OrtNode* node, _Out_ size_t* out);

/** \brief Gets the i-th input name of the node.
 *
 * \param[in] node The node to query
 * \param[in] i The index of the input
 * \param[out] out The i-th input name of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetIthInputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

/** \brief Gets the number of outputs of the node.
 *
 * \param[in] node The node to query
 * \param[out] out The number of outputs of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetNumOutputs, const OrtNode* node, _Out_ size_t* out);

/** \brief Gets the i-th output name of the node.
 *
 * \param[in] node The node to query
 * \param[in] i The index of the output
 * \param[out] out The i-th output name of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetIthOutputName, const OrtNode* node, size_t i, _Outptr_ const char** out);

/** \brief Gets the Node's NodeIndex.
 *
 * \param[in] node The node to query
 * \param[out] out The Node's NodeIndex
 *
 */
ORT_API2_STATUS(OrtNode_GetIndex, const OrtNode* node, _Out_ size_t* out);

/** \brief Gets attribute names of the node.
 *
 * \param[in] node The node to query
 * \param[out] names The attribute names of the node
 * \param[out] num The number of attribute names
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeNames, const OrtNode* node, _Out_ const char*** names, _Out_ size_t* num);

/** \brief Gets the attribute size of the node.
 *
 * \param[in] node The node to query
 * \param[out] out The attribute size of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeSize, const OrtNode* node, _Out_ size_t* out);

/** \brief Gets the attribute type of the node.
 *
 * \param[in] node The node to query
 * \param[in] attribute The attribute name
 * \param[out] out The attribute type of the node
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeType, const OrtNode* node, const char* attribute, _Out_ int* out); // AttributeProto_AttributeType

/** \brief Check if the attribute key exists in the node.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out 1 if the attribute key exists in the node, 0 otherwise
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeKeyCount, const OrtNode* node, const char* key, _Out_ size_t* out);

/** \brief Gets how many ints are in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The number of ints in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeIntSize, const OrtNode* node, const char* key, _Out_ int* out);

/** \brief Gets how many floats are in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The number of floats in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeFloatSize, const OrtNode* node, const char* key, _Out_ int* out);

/** \brief Gets how many strings are in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The number of strings in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeStringSize, const OrtNode* node, const char* key, _Out_ int* out);

/** \brief Gets the i-th int in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[in] i The index of the int
 * \param[out] out The i-th int in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeIthInt, const OrtNode* node, const char* key, int i, _Out_ int64_t* out);

/** \brief Gets the i-th float in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[in] i The index of the float
 * \param[out] out The i-th float in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeIthFloat, const OrtNode* node, const char* key, int i, _Out_ float* out);

/** \brief Gets the i-th string in the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[in] i The index of the string
 * \param[out] out The i-th string in the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeIthStr, const OrtNode* node, const char* key, int i, _Outptr_ const char** out);

/** \brief Gets the string value of the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The string value of the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeStr, const OrtNode* node, const char* key, _Outptr_ const char** out);

/** \brief Gets the int value of the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The int value of the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeInt, const OrtNode* node, const char* key, _Out_ int64_t* out);

/** \brief Gets the float value of the attribute with the given key.
 *
 * \param[in] node The node to query
 * \param[in] key The attribute key
 * \param[out] out The float value of the attribute
 *
 */
ORT_API2_STATUS(OrtNode_GetAttributeFloat, const OrtNode* node, const char* key, _Out_ float* out);

/** \brief Gets the subgraphs of the given node.
 *
 * \param[in] node The node to query
 * \param[out] subgraphs The subgraphs of the node
 * \param[out] num_subgraphs The number of subgraphs
 *
 */
ORT_API2_STATUS(OrtNode_GetSubgraphs, const OrtNode* node, _Outptr_ const OrtGraphViewer*** subgraphs, _Out_ size_t* num_subgraphs);

/** \brief Free the memory
 *
 * \param[in] p The memory to free
 *
 */
ORT_API2_STATUS(OrtFreeMem, void* p);
};
typedef struct OrtGraphApi OrtGraphApi;
