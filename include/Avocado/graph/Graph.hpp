/*
 * Graph.hpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_GRAPH_HPP_
#define AVOCADO_GRAPH_HPP_

#include <Avocado/layers/Layer.hpp>
#include <Avocado/core/DataType.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/graph/GraphNode.hpp>
#include <Avocado/losses/LossFunction.hpp>

#include <memory>
#include <vector>

namespace avocado /* forward declarations */
{
	class Json;
	class Shape;
	class Layer;
	class Device;
	class Tensor;
	class Optimizer;
	class Regularizer;
	namespace inference
	{
		class CalibrationTable;
	}
}

namespace avocado
{
	typedef int GraphNodeID;

	class Graph
	{
		private:
			Context m_context;

			std::vector<std::unique_ptr<Layer>> m_layers;
			std::vector<std::unique_ptr<GraphNode>> m_nodes;
			std::vector<std::unique_ptr<LossFunction>> m_losses;
			std::vector<std::unique_ptr<Tensor>> m_targets;

			std::vector<GraphNode*> m_input_nodes; // non-owning
			std::vector<GraphNode*> m_output_nodes; // non-owning

			std::unique_ptr<Tensor> m_backup_tensor;

			DataType m_datatype = DataType::FLOAT32;

		public:
			Graph(Device device = Device::cpu());

			Graph(const Graph &other) = delete;
			Graph& operator=(const Graph &other) = delete;
			Graph(Graph &&other) = delete;
			Graph& operator=(Graph &&other) = delete;

			Device device() const noexcept;
			DataType dtype() const noexcept;
			const Context& context() const noexcept;

			GraphNodeID addInput(const Shape &shape);
			GraphNodeID add(const Layer &layer, GraphNodeID node);
			GraphNodeID add(const Layer &layer, std::initializer_list<GraphNodeID> nodes);
			void addOutput(GraphNodeID node, const LossFunction &loss);
			void addOutput(GraphNodeID node);

			const Tensor& getInput(int index = 0) const;
			const Tensor& getOutput(int index = 0) const;
			const Tensor& getGradient(int index = 0) const;
			const Tensor& getTarget(int index = 0) const;
			Tensor& getInput(int index = 0);
			Tensor& getOutput(int index = 0);
			Tensor& getGradient(int index = 0);
			Tensor& getTarget(int index = 0);

			Shape getInputShape(int index = 0) const;
			Shape getOutputShape(int index = 0) const;

			int numberOfInputs() const noexcept;
			int numberOfOutputs() const noexcept;
			int maxBatchSize() const;
			void moveTo(Device newDevice);
			void setInputShape(const Shape &shape);
			void setInputShape(const std::vector<Shape> &list);

			void setOptimizer(const Optimizer &optimizer);
			void setRegularizer(const Regularizer &regularizer);
			void init();
			void forward(int batchSize);
			void backward(int batchSize);
			std::vector<Scalar> getLoss(int batchSize);
			void learn();

			void print() const;
			void makeNonTrainable();
			bool isTrainable() const noexcept;
			void calibrate(inference::CalibrationTable &table) const;

			int numberOfLayers() const noexcept;
			const Layer& getLayer(int index) const;
			Layer& getLayer(int index);
			int numberOfNodes() const noexcept;
			const GraphNode& getNode(int index) const;
			GraphNode& getNode(int index);
			GraphNodeID getNodeID(const GraphNode *node) const noexcept;

			void clear();
			Json save(SerializedObject &binary_data) const;
			void load(const Json &json, const SerializedObject &binary_data);

			void insert_node_with_layer(std::unique_ptr<Layer> &&new_layer, const std::vector<GraphNode*> &inputs,
					const std::vector<GraphNode*> &outputs);
			void remove_node(GraphNode *node);
			std::unique_ptr<Layer> replaceLayer(int index, const Layer &newLayer);
		private:
			GraphNodeID add_node(const Layer &layer, const std::vector<GraphNodeID> &inputs);

			void create_backup_tensor();

			Json save_node(const GraphNode *node) const;
			void load_node(const Json &json);
			int index_of_node(const GraphNode *node) const noexcept;
			int index_of_layer(const Layer *layer) const noexcept;

			const GraphNode* get_node(GraphNodeID index) const;
			GraphNode* get_node(GraphNodeID index);
	};

} /* namespace avocado */

#endif /* AVOCADO_GRAPH_HPP_ */
