/*
 * graph_optimizers.hpp
 *
 *  Created on: Feb 28, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_INFERENCE_GRAPH_OPTIMIZERS_HPP_
#define AVOCADO_INFERENCE_GRAPH_OPTIMIZERS_HPP_

#include <Avocado/inference/GraphOptimizer.hpp>

namespace avocado
{
	namespace inference
	{
		class BatchNormToAffine: public GraphOptimizer
		{
			public:
				std::string name() const;
				std::unique_ptr<GraphOptimizer> clone() const;
				bool optimize(Graph &graph) const;
		};

		class MergeActivations: public GraphOptimizer
		{
			public:
				std::string name() const;
				std::unique_ptr<GraphOptimizer> clone() const;
				bool optimize(Graph &graph) const;
		};

		class MergeAffine: public GraphOptimizer
		{
			public:
				std::string name() const;
				std::unique_ptr<GraphOptimizer> clone() const;
				bool optimize(Graph &graph) const;
		};

		class MergeAdd: public GraphOptimizer
		{
			public:
				std::string name() const;
				std::unique_ptr<GraphOptimizer> clone() const;
				bool optimize(Graph &graph) const;
		};

	} /* namespace inference */
} /* namespace avocado */

#endif /* AVOCADO_INFERENCE_GRAPH_OPTIMIZERS_HPP_ */
