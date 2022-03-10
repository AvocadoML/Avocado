/*
 * InputLayer.cpp
 *
 *  Created on: Feb 22, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Input.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/math/activations.hpp>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Input());
	}

	Input::Input(const Shape &input_shape) :
			Layer()
	{
		m_input_shapes.push_back(input_shape);
	}

	void Input::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Input layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Input::getOutputShape() const
	{
		return m_input_shapes[0];
	}

	std::string Input::name() const
	{
		return "Input";
	}
	Json Input::getConfig() const
	{
		Json result = Layer::getConfig();
		result["input_shape"] = getInputShape().toJson();
		return result;
	}

	Input* Input::clone(const Json &config) const
	{
		std::unique_ptr<Input> result = std::make_unique<Input>(Shape(config["input_shape"]));
		result->m_nonlinearity = nonlinearityFromString(config["nonlinearity"]);
		return result.release();
	}

	void Input::forward(const std::vector<Tensor> &input, Tensor &output)
	{
		assert(input.size() == 1);
		assert(same_device(context(), input[0], output));

		math::activationForwardInPlace(context(), m_nonlinearity, output);
	}
	void Input::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar beta)
	{
		assert(input.size() == 1);
		assert(gradientIn.size() == 1);

		math::activationBackwardInPlace(context(), m_nonlinearity, output, gradientOut);
	}

} /* namespace avocado */

