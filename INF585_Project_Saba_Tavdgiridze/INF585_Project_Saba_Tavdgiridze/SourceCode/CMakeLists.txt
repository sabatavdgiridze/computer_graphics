cmake_minimum_required(VERSION 3.0)

project(INF585_Project_Saba_Tavdgiridze)

add_subdirectory(External)

add_executable (
	INF585_Project_Saba_Tavdgiridze
	Main.cpp
)

# Copy the shader files in the binary location. 

add_custom_command(TARGET INF585_Project_Saba_Tavdgiridze 
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:INF585_Project_Saba_Tavdgiridze> ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(INF585_Project_Saba_Tavdgiridze LINK_PRIVATE glad)

target_link_libraries(INF585_Project_Saba_Tavdgiridze LINK_PRIVATE glfw)

target_link_libraries(INF585_Project_Saba_Tavdgiridze LINK_PRIVATE glm)
