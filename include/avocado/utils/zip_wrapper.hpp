/*
 * zip_wrapper.hpp
 *
 *  Created on: Mar 7, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_ZIP_WRAPPER_HPP_
#define AVOCADO_UTILS_ZIP_WRAPPER_HPP_

#include <vector>

namespace avocado
{
	class ZipWrapper
	{
			static const size_t CHUNK = 262144;
		public:
			static std::vector<char> compress(const std::vector<char> &data, int level = -1);
			static std::vector<char> uncompress(const std::vector<char> &data);
	};

} /* namespace avocado */

#endif /* AVOCADO_UTILS_ZIP_WRAPPER_HPP_ */
