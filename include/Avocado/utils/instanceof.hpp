/*
 * instanceof.hpp
 *
 *  Created on: Aug 8, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_INSTANCEOF_HPP_
#define AVOCADO_UTILS_INSTANCEOF_HPP_

namespace avocado
{
	template<typename Base, typename T>
	inline bool instanceof(const T *ptr) noexcept
	{
		return dynamic_cast<const Base*>(ptr) != nullptr;
	}

} /* namespace avocado */

#endif /* AVOCADO_UTILS_INSTANCEOF_HPP_ */
