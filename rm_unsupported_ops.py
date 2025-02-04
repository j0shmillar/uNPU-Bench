import onnx
from onnx import helper, TensorProto

def get_tensor_shape(tensor_name, graph):
    """retrieve the shape of a tensor from graph.value_info or initializers."""
    print(f"searching for tensor: {tensor_name}")
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            print(f"found in value_info: {shape}")
            return shape

    for initializer in graph.initializer:
        if initializer.name == tensor_name:
            shape = list(initializer.dims)
            print(f"found in initializer: {shape}")
            return shape

    print(f"not found in value_info or initializer. checking graph.output...")
    for output in graph.output:
        if output.name == tensor_name:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            print(f"found in graph.output: {shape}")
            return shape

    print(f"tensor {tensor_name} shape not found!")
    return None


def run(input_model_path, output_model_path, unsupported_ops = ['Sigmoid', 'Softmax'], placeholder_shape = [1, 12, 12, 12]):
    model = onnx.load(input_model_path)
    graph = model.graph

    opset_version = model.opset_import[0].version

    nodes_to_remove = set()
    softmax_outputs = set()

    def mark_for_removal(output_names):
        """recursively mark nodes that depend on removed nodes"""
        for node in graph.node:
            if any(inp in output_names for inp in node.input):
                nodes_to_remove.add(node.name)
                softmax_outputs.update(node.output)
                mark_for_removal(node.output)

    for node in graph.node:
        if node.op_type in unsupported_ops:
            nodes_to_remove.add(node.name)
            softmax_outputs.update(node.output)
            mark_for_removal(node.output)
            for input_name in node.input:
                if "Constant" in input_name:
                    nodes_to_remove.add(input_name)
                    for sub_node in graph.node:
                        if input_name in sub_node.output:
                            nodes_to_remove.add(sub_node.name)

    new_nodes = [node for node in graph.node if node.name not in nodes_to_remove]

    orphan_outputs = set()
    all_inputs = {inp for node in new_nodes for inp in node.input}
    for node in new_nodes:
        for output_name in node.output:
            if output_name not in all_inputs:
                orphan_outputs.add(output_name)

    new_outputs = []
    existing_outputs = [output.name for output in graph.output]
    new_output_names = existing_outputs + list(orphan_outputs)
    new_output_names = set(new_output_names)

    # TODO clean
    i = 0
    for output in new_output_names:
        if any(output in node.output for node in new_nodes):
            shape_info = get_tensor_shape(output, graph)
            if not shape_info:
                print("Falling back to placeholder shape") # TODO fix
                shape_info = placeholder_shape
            new_output_name = f"output_{i}"
            new_output = helper.make_tensor_value_info(new_output_name, TensorProto.FLOAT, shape_info)
            new_outputs.append(new_output)
            output_node = helper.make_node(
                "Identity",
                inputs=[output],
                outputs=[new_output_name],
                name=f"Identity:{i}")
            new_nodes.append(output_node)
            i+=1

    new_initializers = [init for init in graph.initializer if init.name not in nodes_to_remove]
    new_value_info = [v for v in graph.value_info if v.name not in nodes_to_remove]

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=graph.name,
        inputs=graph.input,
        outputs=new_outputs,
        initializer=new_initializers,
        value_info=new_value_info)

    new_model = helper.make_model(new_graph, producer_name="modified_model")
    new_model.opset_import[0].version = opset_version

    onnx.save(new_model, output_model_path)
    print(f"updated model saved to: {output_model_path}")
