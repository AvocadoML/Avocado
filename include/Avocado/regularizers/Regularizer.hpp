/*
 * Regularizer.hpp
 *
 *  Created on: Oct 13, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_REGULARIZERS_REGULARIZER_HPP_
#define AVOCADO_REGULARIZERS_REGULARIZER_HPP_

#include <memory>
#include <string>
#include <stdexcept>

namespace avocado /* forward declarations */
{
	class Json;
	class SerializedObject;
	class Parameter;
	class Device;
	class Context;
}

namespace avocado
{

	class Regularizer
	{
		public:
			Regularizer() = default;
			Regularizer(const Regularizer &other) = delete;
			Regularizer(Regularizer &&other) = delete;
			Regularizer& operator=(const Regularizer &other) = delete;
			Regularizer& operator=(Regularizer &&other) = delete;
			virtual ~Regularizer() = default;

			virtual void apply(const Context &context, Parameter &param) = 0;

			virtual std::string name() const = 0;
			virtual Regularizer* clone() const = 0;
			virtual Json serialize(SerializedObject &binary_data) const = 0;
			virtual void unserialize(const Json &json, const SerializedObject &binary_data) = 0;
	};

	void registerRegularizer(const Regularizer &reg);
	std::unique_ptr<Regularizer> loadRegularizer(const Json &json, const SerializedObject &binary_data);

} /* namespace avocado */

#endif /* AVOCADO_REGULARIZERS_REGULARIZER_HPP_ */
