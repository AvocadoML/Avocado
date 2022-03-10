/*
 * Softmax.cpp
 *
 *  Created on: Feb 13, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/layers/Softmax.hpp>
#include <Avocado/core/error_handling.hpp>
#include <Avocado/core/Context.hpp>
#include <Avocado/core/Shape.hpp>
#include <Avocado/core/Tensor.hpp>
#include <Avocado/core/Scalar.hpp>
#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <Avocado/utils/static_block.hpp>

namespace avocado
{
	static_block
	{
//		registerLayer(Softmax());
	}

	Softmax::Softmax(SoftmaxMode mode) :
			m_mode(mode)
	{
		m_nonlinearity = NonlinearityType::SOFTMAX;
	}

	void Softmax::setNonlinearity(NonlinearityType act) noexcept
	{
	}

	void Softmax::setInputShape(const std::vector<Shape> &shapes)
	{
		if (shapes.size() != 1)
			throw IllegalArgument(METHOD_NAME, "Softmax layer expects single input shape");
		m_input_shapes = shapes;
	}
	Shape Softmax::getOutputShape() const
	{
		if (m_input_shapes.size() != 1)
			throw UninitializedObject(METHOD_NAME, "input shape has not been set");
		return m_input_shapes[0];
	}

	std::string Softmax::name() const
	{
		return "Softmax";
	}
	Json Softmax::getConfig() const
	{
		Json result = Layer::getConfig();
		result["mode"] = toString(m_mode);
		return result;
	}

	Softmax* Softmax::clone(const Json &config) const
	{
		return new Softmax(softmaxModeFromString(config["mode"]));
	}

	void Softmax::forward(const std::vector<Tensor> &input, Tensor &output, Scalar alpha, Scalar beta)
	{
		assert(input.size() == 1);
//		math::softmaxForward(context(), m_mode, 1, 0, input[0], output);
	}
	void Softmax::backward(const std::vector<Tensor> &input, const Tensor &output, std::vector<Tensor> &gradientIn, Tensor &gradientOut, Scalar alpha,
			Scalar beta)
	{
		assert(input.size() == 1);
//		math::softmaxBackward(context(), m_mode, 1, 0, gradient_prev[0], gradient_next, output);
	}
} /* namespace avocado */
