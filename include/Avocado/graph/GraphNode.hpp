/*
 * GraphNode.hpp
 *
 *  Created on: Feb 19, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_GRAPH_GRAPHNODE_HPP_
#define AVOCADO_GRAPH_GRAPHNODE_HPP_

#include <Avocado/core/DataType.hpp>
#include <Avocado/core/Shape.hpp>

#include <memory>
#include <vector>

namespace avocado /* forward declarations */
{
	class Json;
	class Shape;
	class Layer;
	class Device;
	class Tensor;
}

namespace avocado
{

	class GraphNode
	{
		private:
			Layer *m_layer = nullptr; // non-owning

			std::vector<GraphNode*> m_input_nodes; // non-owning
			std::vector<GraphNode*> m_output_nodes; // non-owning

			std::unique_ptr<Tensor> m_output_tensor;
			std::unique_ptr<Tensor> m_gradient_tensor;
			Shape m_output_shape;

			bool m_done_backward = false;
			bool m_layer_is_shared = false;
			bool m_is_bypassed_during_backward = false;
		public:
			GraphNode(Layer *layer, const std::vector<GraphNode*> input_nodes);

			bool isInputNode() const noexcept;
			bool isOutputNode() const noexcept;
			Shape getOutputShape() const noexcept;
			void resolveInputShapes();
			int getBackupStorage();

			int numberOfInputs() const noexcept;
			int numberOfOutputs() const noexcept;
			const GraphNode* getInputNode(int index) const;
			GraphNode* getInputNode(int index);
			const GraphNode* getOutputNode(int index) const;
			GraphNode* getOutputNode(int index);
			std::vector<GraphNode*> getInputs() const;
			std::vector<GraphNode*> getOutputs() const;

			void forward(int batch_size);
			void backward(int batch_size, Tensor &backup_tensor);
			void prepareForBackward();

			const Layer& getLayer() const;
			Layer& getLayer();
			const Tensor& getOutputTensor() const;
			Tensor& getOutputTensor();
			const Tensor& getGradientTensor() const;
			Tensor& getGradientTensor();

			void moveTo(Device newDevice);
			void makeNonTrainable() noexcept;
			void bypassDuringBackward() noexcept;

			static void link(GraphNode *prev, GraphNode *next);
			static void link(const std::vector<GraphNode*> &prev, GraphNode *next);
			static void link(GraphNode *prev, const std::vector<GraphNode*> &next);
			static void link(const std::vector<GraphNode*> &prev, const std::vector<GraphNode*> &next);

			static void removeLink(GraphNode *prev, GraphNode *next);
			void removeAllLinks();

			void replaceLayer(Layer *new_layer);
	};

} /* namespace avocado */

#endif /* AVOCADO_GRAPH_GRAPHNODE_HPP_ */
