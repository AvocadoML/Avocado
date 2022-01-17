/*
 * LossFunction.hpp
 *
 *  Created on: Feb 16, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_LOSSES_LOSSFUNCTION_HPP_
#define AVOCADO_LOSSES_LOSSFUNCTION_HPP_

#include <memory>
#include <string>
#include <stdexcept>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Scalar;
	class Tensor;
	class Context;
	class Layer;
}

namespace avocado
{

	class LossFunction
	{
		public:
			LossFunction() = default;
			LossFunction(const LossFunction &other) = delete;
			LossFunction(LossFunction &&other) = delete;
			LossFunction& operator=(const LossFunction &other) = delete;
			LossFunction& operator=(LossFunction &&other) = delete;
			virtual ~LossFunction() = default;

			virtual bool tryCombineWith(const Layer &layer) noexcept;

			virtual Scalar getLoss(const Context &context, const Tensor &output, const Tensor &target) const = 0;
			virtual void getGradient(const Context &context, Tensor &gradient, const Tensor &output, const Tensor &target) const = 0;

			virtual std::string name() const = 0;
			virtual LossFunction* clone() const = 0;
			virtual Json serialize(SerializedObject &binary_data) const;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data);
	};

	void registerLossFunction(const LossFunction &loss);
	std::unique_ptr<LossFunction> loadLossFunction(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_LOSSES_LOSSFUNCTION_HPP_ */
