# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import List, Optional, Tuple, Union

import math
import numpy as np
from fusion_base import Fusion
from fusion_options import AttentionMaskFormat
from fusion_utils import FusionUtils, NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class AttentionMask:
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel):
        self.model = model
        # A lookup table with mask input as key, and mask index output as value
        self.mask_indice = {}
        # A lookup table with mask input as key, and cast (to int32) output as value
        self.mask_casted = {}
        self.utils = FusionUtils(model)
        self.mask_format = AttentionMaskFormat.MaskIndexEnd
        self.opset_version = model.get_opset_version()

    def set_mask_format(self, mask_format: AttentionMaskFormat):
        self.mask_format = mask_format

    def set_mask_indice(self, mask, mask_index):
        if mask in self.mask_indice:
            assert mask_index == self.mask_indice[mask]
        self.mask_indice[mask] = mask_index

    def get_first_mask(self):
        assert len(self.mask_indice) > 0
        return next(iter(self.mask_indice))

    def process_mask(self, input: str) -> str:
        if self.mask_format == AttentionMaskFormat.NoMask:
            return None

        if input in self.mask_indice:
            return self.mask_indice[input]

        # Add cast to convert int64 to int32
        if self.model.find_graph_input(input):
            casted, input_name = self.utils.cast_graph_input_to_int32(input)
        else:
            input_name, cast_node = self.utils.cast_input_to_int32(input)
            casted = True

        if casted:
            self.mask_casted[input] = input_name

        # Attention supports int32 attention mask (2D) since 1.4.0
        if self.mask_format == AttentionMaskFormat.AttentionMask:
            self.mask_indice[input] = input_name
            return input_name

        # Add a mask processing node to convert attention mask to mask index (1D)
        output_name = self.model.create_node_name("mask_index")
        if self.opset_version < 13:
            mask_index_node = helper.make_node(
                "ReduceSum",
                inputs=[input_name],
                outputs=[output_name],
                name=self.model.create_node_name("ReduceSum", "MaskReduceSum"),
            )
            mask_index_node.attribute.extend([helper.make_attribute("axes", [1]), helper.make_attribute("keepdims", 0)])
        else:
            # ReduceSum-13: axes is moved from attribute to input
            axes_name = "ort_const_1_reduce_sum_axes"
            if self.model.get_initializer(axes_name) is None:
                self.model.add_initializer(
                    helper.make_tensor(
                        name=axes_name,
                        data_type=TensorProto.INT64,
                        dims=[1],
                        vals=[1],
                    )
                )
            mask_index_node = helper.make_node(
                "ReduceSum",
                inputs=[input_name, axes_name],
                outputs=[output_name],
                name=self.model.create_node_name("ReduceSum", "MaskReduceSum"),
            )
            mask_index_node.attribute.extend([helper.make_attribute("keepdims", 0)])

        self.model.add_node(mask_index_node)

        self.mask_indice[input] = output_name
        return output_name


class FusionAttention(Fusion):
    """
    Fuse Attention subgraph into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        attention_mask: AttentionMask,
        use_multi_head_attention: bool = False,
        search_op_types: List[str] = ["SkipLayerNormalization", "LayerNormalization"],  # noqa: B006
    ):
        attention_op_name = "MultiHeadAttention" if use_multi_head_attention else "Attention"
        super().__init__(model, attention_op_name, search_op_types)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_mask = attention_mask
        self.use_multi_head_attention = use_multi_head_attention
        self.mask_filter_value = None

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads_and_hidden_size_from_concat(self, concat: NodeProto) -> Tuple[int, int]:
        """
        Detect num_heads and hidden_size from Concat node in the following subgraph:

        SkipLayerNormalization or EmbedLayerNormalization
                        /        |
                     MatMul    Shape
                        |        |
                       Add     Gather(indices=0)
                        |        |
                        |      Unsqueeze
                        |        |
                        |     Concat (*, -1, 12, 64)
                        |     /
                       Reshape
                          |
                       Transpose
        """
        if len(concat.input) == 4:
            num_heads = self.model.get_constant_value(concat.input[2])
            head_size = self.model.get_constant_value(concat.input[3])
            if (
                isinstance(num_heads, np.ndarray)
                and num_heads.size == 1
                and isinstance(head_size, np.ndarray)
                and head_size.size == 1
            ):
                return num_heads[0], num_heads[0] * head_size[0]

        return self.num_heads, self.hidden_size

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto) -> Tuple[int, int]:
        """Detect num_heads and hidden_size from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q

        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        q_shape = self.model.get_initializer(reshape_q.input[1])
        if q_shape is None:
            concat = self.model.get_parent(reshape_q, 1)
            if concat is not None and concat.op_type == "Concat":
                return self.get_num_heads_and_hidden_size_from_concat(concat)
            logger.debug(f"{reshape_q.input[1]} is not initializer.")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        q_shape_value = NumpyHelper.to_array(q_shape)
        if len(q_shape_value) != 4 or (q_shape_value[2] <= 0 or q_shape_value[3] <= 0):
            logger.debug(f"q_shape_value={q_shape_value}. Expected value are like [0, 0, num_heads, head_size].")
            return self.num_heads, self.hidden_size  # Fall back to user specified value

        num_heads = q_shape_value[2]
        head_size = q_shape_value[3]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def get_add_qk_str(self, add_qk: NodeProto):
        shape_infer = self.model.infer_runtime_shape(update=True)
        if shape_infer is None:
            return

        input_0_shape = shape_infer.get_edge_shape(add_qk.input[0])
        input_1_shape = shape_infer.get_edge_shape(add_qk.input[1])

        if input_0_shape is None or input_1_shape is None:
            logger.debug(f"one of the inputs of {add_qk} is None")
            return None

        if input_0_shape != input_1_shape:
            logger.debug(f"the shape of two inputs of {add_qk} is not same")
            return None

        return add_qk.input[1]

    def concat_kv(self, past_k: str, past_v: str) -> str:
        """Concatenate past_k and past_v inputs to create past_kv input.

        Args:
            past_k (str): name of past K value
            past_v (str): name of past V value

        Returns:
            kv_output_name (str): name of past KV value
        """
        # Unsqueeze K and V nodes from (B,N,P,H) to (1,B,N,P,H)
        # B = batch size, N = num heads, P = past sequence length, H = head size
        unsqueeze_k_name = self.model.create_node_name("Unsqueeze")
        unsqueeze_v_name = self.model.create_node_name("Unsqueeze")
        k_5d_name = (past_k + "_5d").replace(".", "_")
        v_5d_name = (past_v + "_5d").replace(".", "_")

        k_5d = helper.make_node(
            "Unsqueeze",
            inputs=[past_k],
            outputs=[k_5d_name],
            name=unsqueeze_k_name,
            axes=[0],
        )
        v_5d = helper.make_node(
            "Unsqueeze",
            inputs=[past_v],
            outputs=[v_5d_name],
            name=unsqueeze_v_name,
            axes=[0],
        )

        # Add unsqueeze nodes to graph
        self.nodes_to_add.append(k_5d)
        self.nodes_to_add.append(v_5d)
        self.node_name_to_graph_name[unsqueeze_k_name] = self.this_graph_name
        self.node_name_to_graph_name[unsqueeze_v_name] = self.this_graph_name

        # Concat K and V to get one node of size (2,B,N,P,H)
        concat_node_name = self.model.create_node_name("Concat")
        kv_output_name = past_v.replace(".value", ".kv").replace(".", "_").replace("_value", "_kv")
        concat_kv = helper.make_node(
            "Concat",
            inputs=[k_5d_name, v_5d_name],
            outputs=[kv_output_name],
            name=concat_node_name,
            axis=0,
        )

        # Add concat node to graph
        self.nodes_to_add.append(concat_kv)
        self.node_name_to_graph_name[concat_node_name] = self.this_graph_name

        return kv_output_name

    def reshape_kv(self, past_k: str, past_v: str) -> (str, str):
        """Reshape past_k and past_v from 4D to 3D to use as inputs for multihead attention node.

        Args:
            past_k (str): name of past K value of shape 4D
            past_v (str): name of past V value of shape 4D

        Returns:
            k_3d (str): name of past K value of shape 3D
            v_3d (str): name of past V value of shape 3D
        """
        # Reshape past_k and past_v from (B,N,P,H) to (B,P,N*H)
        # B = batch size, N = num heads, P = past seq len, H = head size

        # Create initializer for reshaping past_k and past_v
        new_dims_name = "kv_4d_to_3d"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(
                np.array([0, -1, self.model.hidden_size], dtype="int64"), name=new_dims_name
            )
            self.model.add_initializer(new_dims, self.this_graph_name)

        reshape_k_name = self.model.create_node_name("Reshape")
        reshape_v_name = self.model.create_node_name("Reshape")
        k_3d_name = (past_k + "_3d").replace(".", "_")
        v_3d_name = (past_v + "_3d").replace(".", "_")

        k_3d = helper.make_node(
            "Reshape",
            inputs=[past_k, new_dims_name],
            outputs=[k_3d_name],
            name=reshape_k_name,
        )
        v_3d = helper.make_node(
            "Reshape",
            inputs=[past_v, new_dims_name],
            outputs=[v_3d_name],
            name=reshape_v_name,
        )

        # Add reshape nodes to graph
        self.nodes_to_add.append(k_3d)
        self.nodes_to_add.append(v_3d)
        self.node_name_to_graph_name[reshape_k_name] = self.this_graph_name
        self.node_name_to_graph_name[reshape_v_name] = self.this_graph_name

        return k_3d_name, v_3d_name

    def split_kv(self, present_k_name: str, present_v_name: str, kv_node: str):
        """Split kv_node containing present KV values into separate present K and present V values.

        Args:
            present_k_name (str): name of output to store present K value in
            present_v_name (str): name of output to store present V value in
            kv_node (str): name of present KV values
        """
        # Split kv_node into present_k and present_v nodes

        # Create initializers for indexing kv_node, whose shape is (2,B,N,P,H)
        k_index, v_index = "index_0", "index_1"
        k_dim = self.model.get_initializer(k_index)
        v_dim = self.model.get_initializer(v_index)
        if k_dim is None:
            k_dim = numpy_helper.from_array(np.array(0, dtype="int64"), name=k_index)
            self.model.add_initializer(k_dim, self.this_graph_name)
        if v_dim is None:
            v_dim = numpy_helper.from_array(np.array(1, dtype="int64"), name=v_index)
            self.model.add_initializer(v_dim, self.this_graph_name)

        # Create nodes to index kv_node
        gather_k_name = self.model.create_node_name("Gather")
        gather_v_name = self.model.create_node_name("Gather")
        present_k = helper.make_node(
            "Gather",
            inputs=[kv_node, k_index],
            outputs=[present_k_name],
            name=gather_k_name,
            axis=0,
        )
        present_v = helper.make_node(
            "Gather",
            inputs=[kv_node, v_index],
            outputs=[present_v_name],
            name=gather_v_name,
            axis=0,
        )

        # Add gather nodes to graph
        self.nodes_to_add.append(present_k)
        self.nodes_to_add.append(present_v)
        self.node_name_to_graph_name[gather_k_name] = self.this_graph_name
        self.node_name_to_graph_name[gather_v_name] = self.this_graph_name

    def transpose_kv(self, past_k: str, past_v: str):
        """Transpose past_k and past_v from (B,N,P,H) to (B,P,N,H)

        Args:
            past_k (str): name of past K value of shape (B,N,P,H)
            past_v (str): name of past V value of shape (B,N,P,H)

        Returns:
            past_k_transpose (str): name of past K value of shape (B,P,N,H)
            past_v_transpose (str): name of past V value of shape (B,P,N,H)
        """
        past_k_transpose = (past_k + "_transposed").replace(".", "_")
        past_v_transpose = (past_v + "_transposed").replace(".", "_")
        transpose_k_name = self.model.create_node_name("Transpose")
        transpose_v_name = self.model.create_node_name("Transpose")

        transpose_k = helper.make_node(
            "Transpose",
            inputs=[past_k],
            outputs=[past_k_transpose],
            name=transpose_k_name,
            perm=[0, 2, 1, 3],
        )
        transpose_v = helper.make_node(
            "Transpose",
            inputs=[past_v],
            outputs=[past_v_transpose],
            name=transpose_v_name,
            perm=[0, 2, 1, 3],
        )

        # Add reshape nodes to graph
        self.nodes_to_add.append(transpose_k)
        self.nodes_to_add.append(transpose_v)
        self.node_name_to_graph_name[transpose_k_name] = self.this_graph_name
        self.node_name_to_graph_name[transpose_v_name] = self.this_graph_name

        return past_k_transpose, past_v_transpose

    def create_packed_qkv_matmul_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: Union[NodeProto, None],
        v_add: Union[NodeProto, None],
        num_heads: int,
    ) -> Union[NodeProto, None]:
        """Create packed QKV MatMul node before MultiHeadAttention node.
           This is for the scenario where an Attention node should be created but cannot be created
           because past_key and past_value are separate inputs and not one concatenated input.

        Args:
            q_matmul (NodeProto): name of MatMul from Q path - (batch_size, sequence_length, hidden_size)
            k_matmul (NodeProto): name of MatMul from K path - (batch_size, sequence_length, hidden_size)
            v_matmul (NodeProto): name of MatMul from V path - (batch_size, sequence_length, hidden_size)
            q_add (NodeProto): name of Add from Q path
            k_add (NodeProto): name of Add from K path
            v_add (NodeProto): name of Add from V path
            num_heads (int): number of heads

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        matmul_node_name = self.model.create_node_name("MatMul")

        # Check that input for Q, K, V is the same
        assert q_matmul.input[0] == k_matmul.input[0] and k_matmul.input[0] == v_matmul.input[0]

        # Created packed QKV weight
        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        assert qw.shape == kw.shape and kw.shape == vw.shape
        d = qw.shape[0]

        qkv_weight = np.stack((qw, kw, vw), axis=1).reshape((d, 3 * d))
        qkv_weight_name = matmul_node_name + "_qkv_weight"
        weight = helper.make_tensor(
            name=qkv_weight_name,
            data_type=TensorProto.FLOAT,
            dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
            vals=qkv_weight.flatten().tolist(),
        )
        self.model.add_initializer(weight, self.this_graph_name)

        # Created packed QKV MatMul with output (B, S, 3*D)
        # Output is of the form:
        #
        # [[[Q Q ... Q Q K K ... K K V V ... V V]]]
        #   [Q Q ... Q Q K K ... K K V V ... V V]
        #                     .
        #                     .
        #                     .
        #  [[Q Q ... Q Q K K ... K K V V ... V V]
        #   [Q Q ... Q Q K K ... K K V V ... V V]]]
        qkv_matmul_output = matmul_node_name + "_qkv_out"
        qkv_matmul = helper.make_node(
            "MatMul",
            inputs=[q_matmul.input[0], qkv_weight_name],
            outputs=[qkv_matmul_output],
            name=matmul_node_name,
        )
        self.node_name_to_graph_name[matmul_node_name] = self.this_graph_name

        # Create Slice nodes to access Q, K, V
        q_slice_name = matmul_node_name + "_q_start_index"
        q_start_tensor = helper.make_tensor(name=q_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[0])
        k_slice_name = matmul_node_name + "_k_start_index"
        k_start_tensor = helper.make_tensor(name=k_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[d])
        v_slice_name = matmul_node_name + "_v_start_index"
        v_start_tensor = helper.make_tensor(name=v_slice_name, data_type=TensorProto.INT64, dims=[1], vals=[2 * d])
        end_of_qkv_name = matmul_node_name + "_end_of_qkv_index"
        end_of_qkv_tensor = helper.make_tensor(
            name=end_of_qkv_name, data_type=TensorProto.INT64, dims=[1], vals=[3 * d]
        )
        qkv_last_axis_name = matmul_node_name + "_qkv_last_axis"
        qkv_axis_tensor = helper.make_tensor(name=qkv_last_axis_name, data_type=TensorProto.INT64, dims=[1], vals=[-1])

        self.model.add_initializer(q_start_tensor, self.this_graph_name)
        self.model.add_initializer(k_start_tensor, self.this_graph_name)
        self.model.add_initializer(v_start_tensor, self.this_graph_name)
        self.model.add_initializer(end_of_qkv_tensor, self.this_graph_name)
        self.model.add_initializer(qkv_axis_tensor, self.this_graph_name)

        q_slice_output = matmul_node_name + "_q_out"
        q_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, q_slice_name, k_slice_name, qkv_last_axis_name],
            outputs=[q_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[q_slice.name] = self.this_graph_name
        k_slice_output = matmul_node_name + "_k_out"
        k_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, k_slice_name, v_slice_name, qkv_last_axis_name],
            outputs=[k_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[k_slice.name] = self.this_graph_name
        v_slice_output = matmul_node_name + "_v_out"
        v_slice = helper.make_node(
            "Slice",
            inputs=[qkv_matmul_output, v_slice_name, end_of_qkv_name, qkv_last_axis_name],
            outputs=[v_slice_output],
            name=self.model.create_node_name("Slice"),
        )
        self.node_name_to_graph_name[v_slice.name] = self.this_graph_name

        # Add nodes to graph
        self.nodes_to_add.extend([qkv_matmul, q_slice, k_slice, v_slice])
        return q_slice, k_slice, v_slice

    def create_multihead_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: Union[NodeProto, str, None],
        v_matmul: Union[NodeProto, str, None],
        q_add: NodeProto,
        k_add: Union[NodeProto, None],
        v_add: Union[NodeProto, None],
        num_heads: int,
        hidden_size: int,
        output: str,
        key_padding_mask: str = "",
        add_qk: str = "",
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        packed_qkv: bool = False,
    ) -> Union[NodeProto, None]:
        """Create a MultiHeadAttention node.

        Args:
            q_matmul (NodeProto): name of MatMul from Q path - (batch_size, sequence_length, hidden_size)
            k_matmul (NodeProto): name of MatMul from K path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            v_matmul (NodeProto): name of MatMul from V path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            q_add (NodeProto): name of Add from Q path
            k_add (NodeProto): name of Add from K path
            v_add (NodeProto): name of Add from V path
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            output (str): output name of MHA
            key_padding_mask (str): name of key padding mask
            add_qk (str): name of add after Q x K'
            past_k (str): name of past K value - (batch_size, num_heads, past_sequence_length, head_size)
            past_v (str): name of past V value - (batch_size, num_heads, past_sequence_length, head_size)
            present_k (str): name of present K value - (batch_size, num_heads, sequence_length, head_size)
            present_v (str): name of present V value - (batch_size, num_heads, sequence_length, head_size)
            packed_qkv (bool): whether to combine MatMuls from Q, K, V paths
                               Note: This is for the scenario where an Attention node should be created but cannot be created
                               because past_key and past_value are separate inputs and not one concatenated input.

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        # B = batch size, N = num heads, P = past seq len, H = head size
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        graph_input_names = set([node.name for node in self.model.graph().input])
        graph_output_names = set([node.name for node in self.model.graph().output])
        mha_node_name = self.model.create_node_name("Attention")

        # Add initial Q/K/V inputs for MHA
        mha_inputs = []
        if packed_qkv:
            q_slice, k_slice, v_slice = self.create_packed_qkv_matmul_node(
                q_matmul, k_matmul, v_matmul, q_add, k_add, v_add, num_heads
            )
            mha_inputs.extend([q_slice.output[0], k_slice.output[0], v_slice.output[0]])
        elif type(k_matmul) == NodeProto and type(v_matmul) == NodeProto:
            mha_inputs.extend([q_matmul.output[0], k_matmul.output[0], v_matmul.output[0]])
        elif (
            type(k_matmul) == str
            and type(v_matmul) == str
            and k_matmul in graph_input_names
            and v_matmul in graph_input_names
        ):
            mha_inputs.extend([q_matmul.output[0], k_matmul, v_matmul])
        else:
            return None

        # Create combined Q/K/V bias
        q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
        qb = NumpyHelper.to_array(q_bias)
        kb = np.zeros_like(qb)
        vb = np.zeros_like(qb)
        if k_add is not None:
            k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
            kb = NumpyHelper.to_array(k_bias)
        if v_add is not None:
            v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])
            vb = NumpyHelper.to_array(v_bias)

        qkv_bias = np.stack((qb, kb, vb), axis=0)
        qkv_bias_dim = 3 * np.prod(qb.shape)

        bias_name = mha_node_name + "_qkv_bias"
        bias = helper.make_tensor(
            name=bias_name,
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkv_bias.flatten().tolist(),
        )

        # Convert bias to FP16 if model is using FP16
        if q_bias.data_type == 10:
            bias.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(bias).astype(np.float16), bias.name))
        self.model.add_initializer(bias, self.this_graph_name)

        # Add bias to inputs for MHA
        mha_inputs.append(bias_name)

        # Add optional inputs for MHA
        if past_k and past_v and past_k in graph_input_names and past_v in graph_input_names:
            mha_inputs.extend([key_padding_mask, add_qk, past_k, past_v])

        # Add outputs for MHA
        mha_outputs = [output]
        if present_k and present_v and present_k in graph_output_names and present_v in graph_output_names:
            mha_outputs.extend([present_k, present_v])

        mha_node = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=mha_node_name,
        )
        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        return mha_node

    def get_initializer_input_tensor(self, node : NodeProto, convert_to_fp16 : bool = False):
        if node is None:
            return None
        for input_name in node.input:
            ini_tensor = self.model.get_initializer(input_name)
            if ini_tensor:
                if convert_to_fp16 and ini_tensor.data_type != 10:
                    # Convert bias to FP16 if model is using FP16
                    ini_tensor = numpy_helper.from_array(NumpyHelper.to_array(ini_tensor).astype(np.float16), bias.name + "_to_fp16")
                    self.model.add_initializer(ini_tensor, self.this_graph_name)
                return ini_tensor
        raise ValueError(f"Can not found initializer for node: {node}")

    def create_decoder_masked_multihead_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: Union[NodeProto, str, None],
        v_matmul: Union[NodeProto, str, None],
        q_add: NodeProto,
        k_add: Union[NodeProto, None],
        v_add: Union[NodeProto, None],
        num_heads: int,
        hidden_size: int,
        output: str,
        key_padding_mask: str = "",
        add_qk: str = "",
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        past_sequence_length: str = "past_sequence_length",
        beam_width: str = "beam_width",
        cache_indirection: str = "cache_indirection"
    ) -> Union[NodeProto, None]:
        """Create a MultiHeadAttention node.

        Args:
            q_matmul (NodeProto): name of MatMul from Q path - (batch_size, sequence_length, hidden_size)
            k_matmul (NodeProto): name of MatMul from K path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            v_matmul (NodeProto): name of MatMul from V path - (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, past_sequence_length, head_size)
            q_add (NodeProto): name of Add from Q path
            k_add (NodeProto): name of Add from K path
            v_add (NodeProto): name of Add from V path
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            output (str): output name of MHA
            key_padding_mask (str): name of key padding mask
            add_qk (str): name of add after Q x K'
            past_k (str): name of past K value - (batch_size, num_heads, past_sequence_length, head_size)
            past_v (str): name of past V value - (batch_size, num_heads, past_sequence_length, head_size)
            present_k (str): name of present K value - (batch_size, num_heads, sequence_length, head_size)
            present_v (str): name of present V value - (batch_size, num_heads, sequence_length, head_size)
            past_sequence_length, beam_width, cache_indirection : name for the input tensor (maybe graph input)

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        # B = batch size, N = num heads, P = past seq len, H = head size
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        graph_input_names = set([node.name for node in self.model.graph().input])
        graph_output_names = set([node.name for node in self.model.graph().output])
        mha_node_name = self.model.create_node_name("DecoderMaskedMultiHeadAttention")

        mha_inputs = []
        # Add initial Q/K/V inputs for MHA
        if type(k_matmul) == NodeProto and type(v_matmul) == NodeProto:
            mha_inputs.extend([q_matmul.output[0], k_matmul.output[0], v_matmul.output[0]])
        elif (
            type(k_matmul) == str
            and type(v_matmul) == str
            and k_matmul in graph_input_names
            and v_matmul in graph_input_names
        ):
            mha_inputs.extend([q_matmul.output[0], k_matmul, v_matmul])
        else:
            return None

        # Create combined Q/K/V bias
        q_bias = get_initializer_input_tensor(q_add)
        k_bias = get_initializer_input_tensor(k_add)
        v_bias = get_initializer_input_tensor(v_add)
        has_bias  = q_bias is not None and k_bias is not None and v_bias is not None

        # Add optional inputs for MHA
        if past_k and past_v:
            assert past_k in graph_input_names and past_v in graph_input_names
            assert present_k and present_v

        with_past_kv = 1 if past_k and past_v else 0
        if with_past_kv:
            mha_inputs.extend([key_padding_mask, add_qk, past_k, past_v])
            mha_inputs.extend([past_sequence_length, beam_width, cache_indirection])
        elif has_bias:
            mha_inputs.extend(["", "", "", ""])
            mha_inputs.extend(["", "", ""])

        if has_bias:
            mha_inputs.extend(["" if q_bias is None else q_bias.name])
        if k_bias is not None or v_bias is not None:
            mha_inputs.extend(["" if k_bias is None else k_bias.name])
        if v_bias is not None:
            mha_inputs.extend([v_bias.name])

        # Add outputs for MHA
        mha_outputs = [output]
        if with_past_kv:
            mha_outputs.extend([present_k, present_v])

        mha_node = helper.make_node(
            "DecoderMaskedMultiHeadAttention",
            inputs=mha_inputs,
            outputs=mha_outputs,
            name=mha_node_name,
        )
        mha_node.domain = "com.microsoft"
        mha_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])
        mha_node.attribute.extend([helper.make_attribute("past_present_share_buffer", 1 if with_past_kv else 0)])
        if hidden_size > 0:
            mha_node.attribute.extend([helper.make_attribute("scale", 1.0 / math.sqrt(hidden_size / num_heads))])

        return mha_node

    def create_attention_node(
        self,
        mask_index: str,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        q_add: NodeProto,
        k_add: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
        add_qk_str: str = "",
        past_k: str = "",
        past_v: str = "",
        present_k: str = "",
        present_v: str = "",
        scale: Optional[float] = None,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            mask_index (str): mask input
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            q_add (NodeProto): Add bias node in fully connection for Q
            k_add (NodeProto): Add bias node in fully connection for K
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name
            add_qk_str (str): name of Add node after Q x K'
            past_k (str): name of input for past K value
            past_v (str): name of input for past V value
            present_k (str): name of output to store present K value
            present_v (str): name of output to store present V value

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        assert num_heads > 0

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        has_bias = True
        if q_add is None and k_add is None and v_add is None:
            has_bias = False

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])

        q_bias, k_bias, v_bias = None, None, None
        if has_bias:
            q_bias = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
            k_bias = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
            v_bias = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

            if not (k_weight and v_weight and q_bias and k_bias):
                return None

        if q_weight is None:
            print(
                f"{q_matmul.input[1]} is not an initializer. "
                "Please set do_constant_folding=True in torch.onnx.export to unblock attention fusion"
            )
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)

        # assert q and k have same shape as expected
        assert qw.shape == kw.shape

        qw_in_size = qw.shape[0]
        kw_in_size = kw.shape[0]
        vw_in_size = vw.shape[0]

        assert qw_in_size == kw_in_size == vw_in_size

        if hidden_size > 0 and hidden_size != qw_in_size:
            logger.warning(
                f"Input hidden size ({hidden_size}) is not same as weight matrix dimension of q,k,v ({qw_in_size}). "
                "Please provide a correct input hidden size or pass in 0"
            )

        is_qkv_diff_dims = False
        if qw.shape != vw.shape:
            is_qkv_diff_dims = True

        # All the matrices can have the same shape or q, k matrices can have the same shape with v being different
        # For 2d weights, the shapes would be [in_size, out_size].
        # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
        qw_out_size = np.prod(qw.shape[1:])
        kw_out_size = np.prod(kw.shape[1:])
        vw_out_size = np.prod(vw.shape[1:])

        qkv_weight_dim = 0
        if is_qkv_diff_dims:
            qkv_weight = np.concatenate((qw, kw, vw), axis=1)
            qkv_weight_dim = qw_out_size + kw_out_size + vw_out_size
        else:
            qkv_weight = np.stack((qw, kw, vw), axis=1)
            qkv_weight_dim = 3 * qw_out_size

        if has_bias:
            qb = NumpyHelper.to_array(q_bias)
            kb = NumpyHelper.to_array(k_bias)
            vb = NumpyHelper.to_array(v_bias)

            q_bias_shape = np.prod(qb.shape)
            k_bias_shape = np.prod(kb.shape)
            v_bias_shape = np.prod(vb.shape)

            assert q_bias_shape == k_bias_shape == qw_out_size
            assert v_bias_shape == vw_out_size

            qkv_bias_dim = 0
            if is_qkv_diff_dims:
                qkv_bias = np.concatenate((qb, kb, vb), axis=0)
                qkv_bias_dim = q_bias_shape + k_bias_shape + v_bias_shape
            else:
                qkv_bias = np.stack((qb, kb, vb), axis=0)
                qkv_bias_dim = 3 * q_bias_shape

        attention_node_name = self.model.create_node_name("Attention")

        if not self.use_multi_head_attention:
            weight = helper.make_tensor(
                name=attention_node_name + "_qkv_weight",
                data_type=TensorProto.FLOAT,
                dims=[qw_in_size, qkv_weight_dim],
                vals=qkv_weight.flatten().tolist(),
            )

            # Sometimes weights and bias are stored in fp16
            if q_weight.data_type == 10:
                weight.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(weight).astype(np.float16), weight.name))
            self.model.add_initializer(weight, self.this_graph_name)

        bias = None
        if has_bias:
            bias = helper.make_tensor(
                name=attention_node_name + "_qkv_bias",
                data_type=TensorProto.FLOAT,
                dims=[qkv_bias_dim],
                vals=qkv_bias.flatten().tolist(),
            )
            if q_bias.data_type == 10:
                bias.CopyFrom(numpy_helper.from_array(NumpyHelper.to_array(bias).astype(np.float16), bias.name))
            self.model.add_initializer(bias, self.this_graph_name)

        # For MultiHeadAttention operator, use separated inputs for query, key and value, and no weights.
        if self.use_multi_head_attention:
            if add_qk_str is not None:
                logger.debug("MultiHeadAttention does not support relative_position_bias: cannot fuse the attention.")
                return None

            attention_inputs = [
                q_matmul.output[0],
                k_matmul.output[0],
                v_matmul.output[0],
                attention_node_name + "_qkv_bias",
            ]

            if mask_index is not None:
                attention_inputs.append(mask_index)

            attention_node = helper.make_node(
                "MultiHeadAttention",
                inputs=attention_inputs,
                outputs=[output],
                name=attention_node_name,
            )
        else:
            attention_inputs = [
                input,
                attention_node_name + "_qkv_weight",
                attention_node_name + "_qkv_bias" if has_bias else "",
            ]
            if mask_index is not None:
                attention_inputs.append(mask_index)
            else:
                attention_inputs.append("")

            past_exists = past_k and past_v
            if past_exists:
                past_kv = self.concat_kv(past_k, past_v)
                attention_inputs.append(past_kv)

            if add_qk_str is not None:
                # Convert 4d mask from (B,1,M,M) to (B,N,M,M)
                # B = batch size, M = max sequence length, N = num heads
                concat_node_name = self.model.create_node_name("Concat")
                mask_output_name = add_qk_str + "_mask"
                concat_add_qk_fp32 = helper.make_node(
                    "Concat",
                    inputs=[add_qk_str for _ in range(num_heads)],
                    outputs=[mask_output_name],
                    name=concat_node_name,
                    axis=1,
                )
                # Add new nodes to graph
                self.nodes_to_add.append(concat_add_qk_fp32)
                self.node_name_to_graph_name[concat_node_name] = self.this_graph_name

                # Add attention mask to attention node
                if not past_exists:
                    attention_inputs.append("")
                attention_inputs.append(mask_output_name)

            attention_outputs = [output]
            if present_k and present_v:
                present_kv = present_k.replace(".key", "").replace("_key", "").replace(".", "_")
                attention_outputs.append(present_kv)
                self.split_kv(present_k, present_v, present_kv)

            attention_node = helper.make_node(
                "Attention",
                inputs=attention_inputs,
                outputs=attention_outputs,
                name=attention_node_name,
            )

        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        if scale is not None:
            attention_node.attribute.extend([helper.make_attribute("scale", scale)])

        if is_qkv_diff_dims:
            attention_node.attribute.extend(
                [helper.make_attribute("qkv_hidden_sizes", [qw_out_size, kw_out_size, vw_out_size])]
            )

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == "LayerNormalization":
            add_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(
            start_node,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, 0, 0, 0],
        )
        einsum_node = None
        if qkv_nodes is not None:
            (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes
        else:
            # Match Albert
            qkv_nodes = self.model.match_parent_path(
                start_node, ["Add", "Einsum", "Transpose", "MatMul"], [1, None, 0, 0]
            )
            if qkv_nodes is not None:
                (_, einsum_node, transpose_qkv, matmul_qkv) = qkv_nodes
            else:
                return

        other_inputs = []
        for _i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue

            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        """
        Match flaubert                     Mask
                                            |
        Mul --> LayerNormalization -->  Attention --> MatMul --> Add
         |                                                        |
         |                                                        |
         +---------------------------------------------------------
        """
        mul_before_layernorm = self.model.match_parent(start_node, "Mul", 0)
        if mul_before_layernorm is not None:
            mul_children = input_name_to_nodes[mul_before_layernorm.output[0]]
            if mul_children is not None and len(mul_children) == 2:
                layernorm_node = mul_children[1]
                if layernorm_node.op_type == "LayerNormalization":
                    root_input = layernorm_node.output[0]
                else:
                    return
            elif mul_children is not None and len(mul_children) == 5:
                root_input = mul_before_layernorm.output[0]
            else:
                return
        elif normalize_node.op_type == "LayerNormalization":
            children = input_name_to_nodes[root_input]
            for child in children:
                if child.op_type == "LayerNormalization":
                    root_input = child.output[0]

        """
        When Add before the LayerNormalization produces an output
        that is consumed by some other nodes other than the LayerNormalization itself,
        fused SkipLayerNormalization will have several outputs.
        In this case we need to pick the one used in Attention

        For example, this is the case for ViT

        SkipLayerNormalization --> Attention --> MatMul --> Add --> SkipLayerNormalization
         |                                                                     |
         |                                                                     |
         +---------------------------------------------------------------------+
        """
        parent_node = output_name_to_node[root_input]
        if parent_node.op_type == "SkipLayerNormalization" and len(parent_node.output) == 4:
            root_input = parent_node.output[0]

        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count("MatMul") != 3:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, _, add_v, matmul_v) = v_nodes

        is_distill = False
        is_distill_add = False
        is_no_mask_attention = False
        qk_paths = {
            "path1": (["Softmax", "Add", "Div", "MatMul"], [0, 0, None, 0]),
            "path2": (["Softmax", "Add", "Mul", "MatMul"], [0, 0, None, 0]),
            "path3": (["Softmax", "Where", "MatMul", "Div"], [0, 0, 2, 0]),
            "path4": (["Softmax", "Add", "Where", "MatMul"], [0, 0, 0, 2]),
            "path5": (["Softmax", "Div", "MatMul"], [0, 0, 0]),
        }

        qk_nodes = None
        for k, v in qk_paths.items():
            qk_nodes = self.model.match_parent_path(matmul_qkv, v[0], v[1])
            if qk_nodes is None:
                continue
            if k == "path3":
                is_distill = True
            if k == "path4":
                is_distill_add = True
            if k == "path5":
                is_no_mask_attention = True
            break

        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        add_qk = None
        matmul_qk = None
        where_qk = None
        if is_distill:
            (_, where_qk, matmul_qk, _) = qk_nodes
        elif is_distill_add:
            (_, add_qk, where_qk, matmul_qk) = qk_nodes
        elif is_no_mask_attention:
            (_, _, matmul_qk) = qk_nodes
        else:
            (_, add_qk, _, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, None])
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Div", "Transpose", "Reshape", "Add", "MatMul"],
                [0, 0, 0, 0, None],
            )
            if q_nodes is None:
                logger.debug("fuse_attention: failed to match q path")
                return
        reshape_q = q_nodes[-3]
        add_q = q_nodes[-2]
        matmul_q = q_nodes[-1]

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if k_nodes is None:
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Transpose", "Reshape", "Add", "MatMul"],
                [1, 0, 0, 0, None],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
        add_k = k_nodes[-2]
        matmul_k = k_nodes[-1]

        # Note that Cast might be removed by OnnxRuntime so we match two patterns here.
        mask_nodes = None
        add_qk_str = None
        if is_distill:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Expand", "Reshape", "Equal"], [0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                    (["Cast", "Expand", "Reshape", "Equal"], [0, 0, 0, 0]),
                ],
                output_name_to_node,
            )
        elif is_distill_add:
            _, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (["Cast", "Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0, 0]),
                    (["Equal", "Unsqueeze", "Unsqueeze"], [0, 0, 0]),
                ],
                output_name_to_node,
            )
            if add_qk is not None:
                add_qk_str = self.get_add_qk_str(add_qk)
                if add_qk_str is None:
                    logger.debug(f"fuse_attention: failed to verify shape inference of {add_qk}")
                    return
        elif is_no_mask_attention:
            pass
        else:
            _, mask_nodes, _ = self.model.match_parent_paths(
                add_qk,
                [
                    (
                        ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
                        [None, 0, 1, 0, 0],
                    ),
                    (["Mul", "Sub", "Unsqueeze", "Unsqueeze"], [None, 0, 1, 0]),
                ],
                output_name_to_node,
            )
        if not is_no_mask_attention and mask_nodes is None:
            logger.debug("fuse_attention: failed to match mask path")
            return

        if not is_no_mask_attention and len(mask_nodes) > 1 and mask_nodes[0].op_type == "Mul":
            _, mul_val = self.model.get_constant_input(mask_nodes[0])
            if mul_val != -10000:
                self.mask_filter_value = mul_val

        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_k.input[0] == root_input:
            mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0]) if not is_no_mask_attention else None

            attention_last_node = reshape_qkv if einsum_node is None else transpose_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            # the input_hidden_size represents the input hidden size, this is used as needed but hidden sizes for Q, K are extracted appropriately
            new_node = self.create_attention_node(
                mask_index,
                matmul_q,
                matmul_k,
                matmul_v,
                add_q,
                add_k,
                add_v,
                q_num_heads,
                q_hidden_size,
                root_input,
                attention_last_node.output[0],
                add_qk_str,
            )
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            if einsum_node is not None:
                unique_index = einsum_node.input[0]
                new_edge = "edge_modified_" + unique_index
                shape_tensor = helper.make_tensor(
                    name="shape_modified_tensor" + unique_index,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=np.int64([0, 0, q_num_heads, int(q_hidden_size / q_num_heads)]).tobytes(),
                    raw=True,
                )
                self.model.add_initializer(shape_tensor, self.this_graph_name)
                self.model.add_node(
                    helper.make_node(
                        "Reshape",
                        [attention_last_node.output[0], shape_tensor.name],
                        [new_edge],
                        "reshape_modified_" + unique_index,
                    ),
                    self.this_graph_name,
                )
                einsum_node.input[0] = new_edge

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)

            # For MultiHeadAttention operator, MatMul nodes for Q/K/V projection shall not be fused.
            self.nodes_to_remove.extend(q_nodes if not self.use_multi_head_attention else q_nodes[:-1])
            self.nodes_to_remove.extend(k_nodes if not self.use_multi_head_attention else k_nodes[:-1])
            self.nodes_to_remove.extend(v_nodes if not self.use_multi_head_attention else v_nodes[:-1])

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.prune_graph = True
