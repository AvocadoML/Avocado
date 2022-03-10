/*
 * Layer.hpp
 *
 *  Created on: Oct 16, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LAYERS_LAYER_HPP_
#define AVOCADO_LAYERS_LAYER_HPP_

#include <Avocado/core/DataType.hpp>
#include <Avocado/layers/Parameter.hpp>
#include <Avocado/math/activations.hpp>

#include <memory>
#include <vector>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Context;
	class Shape;
	class Device;
	class Optimizer;
}

namespace avocado
{

	class Layer
	{
		protected:
			Context *m_context = nullptr; // non-owning
			std::vector<Shape> m_input_shapes;

			std::unique_ptr<Parameter> m_weights;
			std::unique_ptr<Parameter> m_bias;

			DataType m_dtype = DataType::FLOAT32;
			NonlinearityType m_nonlinearity;

			size_t m_id;

		public:
			Layer(const std::string &activation = "linear");

			Layer(const Layer &other) = delete;
			Layer(Layer &&other) = delete;
			Layer& operator=(const Layer &other) = delete;
			Layer& operator=(const Layer &&other) = delete;
			virtual ~Layer() = default;

			virtual NonlinearityType getNonlinearity() const noexcept;
			virtual void setNonlinearity(NonlinearityType act) noexcept;
			virtual std::string name() const = 0;
			virtual Json getConfig() const;

			virtual Layer* clone(const Json &config) const = 0;
			virtual Json saveParameters(SerializedObject &binary_data) const;
			virtual void loadParameters(const Json &json, const SerializedObject &binary_data);

			int numberOfInputs() const noexcept;
			void setInputShape(const Shape &shape);
			virtual void setInputShape(const std::vector<Shape> &shapes);
			Shape getInputShape(int index = 0) const;
			virtual Shape getOutputShape() const = 0;
			virtual Shape getWeightShape() const;
			virtual Shape getBiasShape() const;

			Device device() const;
			DataType dtype() const noexcept;
			const Context& context() const;

			Parameter& getWeights();
			Parameter& getBias();
			const Parameter& getWeights() const;
			const Parameter& getBias() const;

			virtual void changeContext(Context &context);

			virtual void init();
			virtual Layer& setInitializer(const Initializer &initializer);
			virtual Layer& setOptimizer(const Optimizer &optimizer);
			virtual Layer& setRegularizer(const Regularizer &regularizer);

			virtual void forward(const std::vector<Tensor> &input, Tensor &output) = 0;
			virtual void backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut,
					Scalar beta) = 0;

			virtual void learn();

			friend bool sameId(const Layer &lhs, const Layer &rhs) noexcept;
	};

	void registerLayer(const Layer &layer);
	std::unique_ptr<Layer> loadLayer(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_LAYERS_LAYER_HPP_ */
