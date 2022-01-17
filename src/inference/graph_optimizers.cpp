/*
 * graph_optimizers.cpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/inference/graph_optimizers.hpp>
#include <Avocado/graph/Graph.hpp>
#include <Avocado/graph/GraphNode.hpp>
#include <Avocado/layers/Activation.hpp>
#include <Avocado/layers/Conv2D.hpp>
#include <Avocado/layers/Affine.hpp>
#include <Avocado/layers/Dense.hpp>
#include <Avocado/layers/Add.hpp>
#include <Avocado/layers/BatchNormalization.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/utils/static_block.hpp>

#include <memory>
#include <string>

namespace
{
	using namespace avocado;
	bool can_merge_activations(const GraphNode *prev, const GraphNode *next)
	{
		return (prev->getLayer().getNonlinearity() == NonlinearityType::LINEAR)
				|| (prev->getLayer().getNonlinearity() == NonlinearityType::RELU && next->getLayer().getNonlinearity() == NonlinearityType::RELU);
	}
}

namespace avocado
{
	namespace inference
	{
		std::string BatchNormToAffine::name() const
		{
			return "BatchNormToAffine";
		}
		std::unique_ptr<GraphOptimizer> BatchNormToAffine::clone() const
		{
			return std::make_unique<BatchNormToAffine>();
		}
		bool BatchNormToAffine::optimize(Graph &graph) const
		{
			static BatchNormalization batchnorm;
			bool has_anything_changed = false;
			for (int i = 0; i < graph.numberOfLayers(); i++)
				if (graph.getLayer(i).name() == batchnorm.name())
				{
					auto old_layer = graph.replaceLayer(i, Affine(toString(graph.getLayer(i).getNonlinearity())));

					const Tensor &batchnorm_weights = old_layer->getWeights().getParam();
					const Tensor &batchnorm_bias = old_layer->getBias().getParam();

					Tensor &affine_weights = graph.getLayer(i).getWeights().getParam();
					Tensor &affine_bias = graph.getLayer(i).getBias().getParam();

					for (int j = 0; j < batchnorm_weights.lastDim(); j++)
					{
						float scale = batchnorm_weights.get<float>( { 1, j }) / batchnorm_weights.get<float>( { 0, j });
						float shift = batchnorm_bias.get<float>( { 1, j }) - scale * batchnorm_bias.get<float>( { 0, j });
						affine_weights.set(scale, { j });
						affine_bias.set(shift, { j });
					}
					has_anything_changed = true;
				}
			return has_anything_changed;
		}

		std::string MergeActivations::name() const
		{
			return "MergeActivations";
		}
		std::unique_ptr<GraphOptimizer> MergeActivations::clone() const
		{
			return std::make_unique<MergeActivations>();
		}
		bool MergeActivations::optimize(Graph &graph) const
		{
			static Activation activation;
			bool has_anything_changed = false;
			for (int i = 0; i < graph.numberOfNodes(); i++)
				if (graph.getNode(i).getLayer().name() == activation.name())
				{
					GraphNode *next = &(graph.getNode(i));
					GraphNode *prev = next->getInputNode(0);
					if (can_merge_activations(prev, next))
					{
						prev->getLayer().setNonlinearity(next->getLayer().getNonlinearity());
						GraphNode::link(prev, next->getOutputs());
						graph.remove_node(next);
						has_anything_changed = true;
					}
				}
			return has_anything_changed;
		}

		std::string MergeAffine::name() const
		{
			return "MergeAffine";
		}
		std::unique_ptr<GraphOptimizer> MergeAffine::clone() const
		{
			return std::make_unique<MergeAffine>();
		}
		bool MergeAffine::optimize(Graph &graph) const
		{
			static Conv2D conv2d(0, 0);
			static Dense dense(0);
			static Affine affine;
			bool has_anything_changed = false;
			for (int i = 0; i < graph.numberOfNodes(); i++)
				if (graph.getNode(i).getLayer().name() == affine.name())
				{
					GraphNode *next = &(graph.getNode(i));
					GraphNode *prev = next->getInputNode(0); // Affine can have only one input
					if (can_merge_activations(prev, next))
					{
						if (prev->getLayer().name() == conv2d.name())
							static_cast<Conv2D&>(prev->getLayer()).useBias(true);
						if (prev->getLayer().name() == dense.name())
							static_cast<Dense&>(prev->getLayer()).useBias(true);

						if (prev->getLayer().name() == conv2d.name() || prev->getLayer().name() == dense.name())
						{
							const int first_dim = prev->getLayer().getWeightShape().firstDim();
							const int last_dim = prev->getLayer().getWeightShape().volumeWithoutFirstDim();
							Tensor weight = prev->getLayer().getWeights().getParam().view( { first_dim, last_dim });

							for (int j = 0; j < first_dim; j++)
							{
								float scale = next->getLayer().getWeights().getParam().get<float>( { j });
								float shift = next->getLayer().getBias().getParam().get<float>( { j });
								for (int k = 0; k < last_dim; k++)
									weight.set(weight.get<float>( { j, k }) * scale, { j, k });
								prev->getLayer().getBias().getParam().set(prev->getLayer().getBias().getParam().get<float>( { j }) * scale + shift, {
										j });
							}
							prev->getLayer().setNonlinearity(next->getLayer().getNonlinearity());

							GraphNode::link(prev, next->getOutputs());
							graph.remove_node(next);
							has_anything_changed = true;
						}
					}
				}
			return has_anything_changed;
		}

		std::string MergeAdd::name() const
		{
			return "MergeAdd";
		}
		std::unique_ptr<GraphOptimizer> MergeAdd::clone() const
		{
			return std::make_unique<MergeAdd>();
		}
		bool MergeAdd::optimize(Graph &graph) const
		{
			static Conv2D conv2d(0, 0);
			static Dense dense(0);
			static Add add_layer;
			bool has_anything_changed = false;
			for (int i = 0; i < graph.numberOfNodes(); i++)
				if (graph.getNode(i).getLayer().name() == add_layer.name() && graph.getNode(i).numberOfInputs() == 2)
				{
					GraphNode *next = &(graph.getNode(i));
					GraphNodeID input_index = -1;
					if (next->getInputNode(0)->getLayer().name() == conv2d.name() || next->getInputNode(0)->getLayer().name() == dense.name())
						input_index = std::max(input_index, graph.getNodeID(next->getInputNode(0)));
					if (next->getInputNode(1)->getLayer().name() == conv2d.name() || next->getInputNode(1)->getLayer().name() == dense.name())
						input_index = std::max(input_index, graph.getNodeID(next->getInputNode(1)));

					if (input_index != -1 && !next->isOutputNode())
					{
						GraphNode *prev = &(graph.getNode(input_index));
						if (can_merge_activations(prev, next))
						{
							prev->getLayer().setNonlinearity(next->getLayer().getNonlinearity());
							GraphNode::removeLink(prev, next);
							GraphNode::link(prev, next->getOutputs());
							GraphNode::link(next->getInputNode(0), prev); // only one input of Add is left now
							graph.remove_node(next);
							has_anything_changed = true;
						}
					}
				}
			return has_anything_changed;
		}

	} /* namespace inference */
} /* namespace avocado */

