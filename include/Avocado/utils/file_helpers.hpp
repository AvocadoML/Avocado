/*
 * file_helpers.hpp
 *
 *  Created on: Jun 17, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_UTILS_FILE_HELPERS_HPP_
#define AVOCADO_UTILS_FILE_HELPERS_HPP_

#include <Avocado/utils/json.hpp>
#include <Avocado/utils/serialization.hpp>
#include <stddef.h>
#include <fstream>
#include <string>

namespace avocado
{
	class FileSaver
	{
			std::string path;
			std::ofstream stream;
		public:
			FileSaver(const std::string &path);
			std::string getPath() const;
			void save(const Json &json, const SerializedObject &binary_data, int indent = -1, bool compress = false);
			void close();
	};

	class FileLoader
	{
			Json json;
			SerializedObject binary_data;
			std::vector<char> loaded_data;
			size_t split_point;
		public:
			FileLoader(const std::string &path, bool uncompress = false);
			const Json& getJson() const noexcept;
			Json& getJson() noexcept;
			const SerializedObject& getBinaryData() const noexcept;
			SerializedObject& getBinaryData() noexcept;
		private:
			void load_all_data();
			size_t find_split_point() const noexcept;
	};

} /* namespace avocado */

#endif /* AVOCADO_UTILS_FILE_HELPERS_HPP_ */
