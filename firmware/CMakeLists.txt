cmake_minimum_required(VERSION 3.13)

set(PICO_BOARD pico)
set(PICO_FLASH_SIZE_BYTES 4194304)

# Point PICO_SDK_PATH at your pico-sdk checkout (env var wins).
if(DEFINED ENV{PICO_SDK_PATH})
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
elseif(NOT DEFINED PICO_SDK_PATH)
    set(PICO_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../pico-sdk)
endif()

include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(picotales C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---- picotales_oled_test — SSD1306 wiring self-test (no model payload) ----
add_executable(picotales_oled_test oled_test.c)
target_link_libraries(picotales_oled_test pico_stdlib)
pico_enable_stdio_usb(picotales_oled_test 1)
pico_enable_stdio_uart(picotales_oled_test 0)
pico_add_extra_outputs(picotales_oled_test)

# ---- picotales_wire_test — board/wire/breadboard continuity tester ----
add_executable(picotales_wire_test wire_test.c)
target_link_libraries(picotales_wire_test pico_stdlib)
pico_enable_stdio_usb(picotales_wire_test 1)
pico_enable_stdio_uart(picotales_wire_test 0)
pico_add_extra_outputs(picotales_wire_test)

# ---- picotales_mic_test — INMP441 I2S capture test, level meter, enrollment
# streamer (+ OLED word echo). Shares the exact 4x-gain + 20 Hz DC-block
# front-end with the voice firmware: enrollment audio must match live audio. ----
add_executable(picotales_mic_test mic_test.c ssd1306.c)
pico_generate_pio_header(picotales_mic_test ${CMAKE_CURRENT_LIST_DIR}/inmp441.pio)
target_link_libraries(picotales_mic_test pico_stdlib hardware_pio hardware_i2c)
pico_enable_stdio_usb(picotales_mic_test 1)
pico_enable_stdio_uart(picotales_mic_test 0)
pico_add_extra_outputs(picotales_mic_test)

# ---- embedded LLM payload (ChinnaLLM-2.7M chat, all-Q8) ----
set(NEO_MODEL_BIN "${CMAKE_CURRENT_LIST_DIR}/../models/chinnallm_chat_q8.bin" CACHE FILEPATH "SPEC_Q4_NEO model binary to embed")
option(NEO_CHAT_MODE "Build with System/User/Assistant chat prompt template" ON)
get_filename_component(NEO_MODEL_BIN_ABS "${NEO_MODEL_BIN}" ABSOLUTE)
string(REGEX REPLACE "\.bin$" "_tok.bin" NEO_TOKENIZER_BIN "${NEO_MODEL_BIN_ABS}")
if(NOT EXISTS "${NEO_MODEL_BIN_ABS}")
    message(FATAL_ERROR "NEO_MODEL_BIN does not exist: ${NEO_MODEL_BIN_ABS}")
endif()
if(NOT EXISTS "${NEO_TOKENIZER_BIN}")
    message(FATAL_ERROR "Neo tokenizer sidecar does not exist: ${NEO_TOKENIZER_BIN}")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import struct,sys; data=open(sys.argv[1],'rb').read(64); vals=struct.unpack('<Ii8iIff3i', data); print(';'.join(map(str, vals[:11] + vals[13:])))"
            "${NEO_MODEL_BIN_ABS}"
    OUTPUT_VARIABLE NEO_HEADER_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE ";" ";" NEO_HEADER_LIST "${NEO_HEADER_RAW}")
list(GET NEO_HEADER_LIST 0 NEO_MODEL_MAGIC)
list(GET NEO_HEADER_LIST 1 NEO_MODEL_VERSION)
list(GET NEO_HEADER_LIST 2 NEO_MODEL_DIM)
list(GET NEO_HEADER_LIST 3 NEO_MODEL_HIDDEN_DIM)
list(GET NEO_HEADER_LIST 4 NEO_MODEL_LAYERS)
list(GET NEO_HEADER_LIST 5 NEO_MODEL_HEADS)
list(GET NEO_HEADER_LIST 6 NEO_MODEL_KV_HEADS)
list(GET NEO_HEADER_LIST 7 NEO_MODEL_VOCAB)
list(GET NEO_HEADER_LIST 8 NEO_MODEL_SEQ_LEN)
list(GET NEO_HEADER_LIST 9 NEO_MODEL_GROUP_SIZE)
list(GET NEO_HEADER_LIST 10 NEO_MODEL_FLAGS)
list(GET NEO_HEADER_LIST 11 NEO_MODEL_EXTRA)
if(NOT NEO_MODEL_MAGIC STREQUAL "1362381901")
    message(FATAL_ERROR "Unsupported neo model magic ${NEO_MODEL_MAGIC}")
endif()
if(NOT NEO_MODEL_VERSION STREQUAL "1")
    message(FATAL_ERROR "Unsupported neo model version ${NEO_MODEL_VERSION}")
endif()
if(NOT NEO_MODEL_EXTRA STREQUAL "4" AND NOT NEO_MODEL_EXTRA STREQUAL "5")
    message(FATAL_ERROR "Unsupported neo version_extra ${NEO_MODEL_EXTRA}")
endif()
math(EXPR NEO_MODEL_HEAD_DIM "${NEO_MODEL_DIM} / ${NEO_MODEL_HEADS}")
math(EXPR NEO_MODEL_KV_DIM "${NEO_MODEL_HEAD_DIM} * ${NEO_MODEL_KV_HEADS}")

set(MODEL_BIN_ABS "${NEO_MODEL_BIN_ABS}")
configure_file(${CMAKE_CURRENT_LIST_DIR}/model_data_q4.S.in ${CMAKE_CURRENT_BINARY_DIR}/model_data_neo.S @ONLY)
set(TOKENIZER_BIN_ABS "${NEO_TOKENIZER_BIN}")
configure_file(${CMAKE_CURRENT_LIST_DIR}/tokenizer_data_q4.S.in ${CMAKE_CURRENT_BINARY_DIR}/tokenizer_data_neo.S @ONLY)

# ---- picotales_chat — serial chat with the on-board LLM (no mic needed) ----
add_executable(picotales_chat
    main_chat_neo.c
    runq4_neo_host.c
    ssd1306.c
    ${CMAKE_CURRENT_BINARY_DIR}/model_data_neo.S
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizer_data_neo.S
)
target_compile_definitions(picotales_chat PRIVATE
    PICO_ON_DEVICE=1
    PICO
    NO_MAIN
    NEO_SEQ_CAP=88
    Q4_MAX_DIM=${NEO_MODEL_DIM}
    Q4_MAX_HIDDEN_DIM=${NEO_MODEL_HIDDEN_DIM}
    Q4_MAX_LAYERS=${NEO_MODEL_LAYERS}
    Q4_MAX_HEADS=${NEO_MODEL_HEADS}
    Q4_MAX_KV_DIM=${NEO_MODEL_KV_DIM}
    Q4_MAX_VOCAB=${NEO_MODEL_VOCAB}
    Q4_MAX_HEAD_DIM=${NEO_MODEL_HEAD_DIM}
    Q4_MAX_SEQ_LEN=${NEO_MODEL_SEQ_LEN}
)
if(NEO_CHAT_MODE)
    target_compile_definitions(picotales_chat PRIVATE NEO_CHAT_MODE)
endif()
target_link_libraries(picotales_chat
    pico_stdlib hardware_pio hardware_vreg hardware_clocks hardware_i2c
)
pico_enable_stdio_usb(picotales_chat 1)
pico_enable_stdio_uart(picotales_chat 0)
pico_add_extra_outputs(picotales_chat)
add_custom_command(TARGET picotales_chat POST_BUILD
    COMMAND arm-none-eabi-size $<TARGET_FILE:picotales_chat>
    COMMENT "Chat firmware sizes:"
)

# ---- picotales_voice_story — the machine: speak 3 words -> story on OLED ----
# Requires vendored TinySR: run tools/tinysr_setup.sh once first.
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/../tinysr/tinysr.c)
set(TINYSR_MODEL_ABS ${CMAKE_CURRENT_LIST_DIR}/../models/tinysr_3word.model)
configure_file(${CMAKE_CURRENT_LIST_DIR}/tinysr_model_data.S.in ${CMAKE_CURRENT_BINARY_DIR}/tinysr_model_data.S @ONLY)
add_executable(picotales_voice_story
    main_voice.c
    runq4_neo_host.c
    ssd1306.c
    ${CMAKE_CURRENT_LIST_DIR}/../tinysr/tinysr.c
    ${CMAKE_CURRENT_BINARY_DIR}/tinysr_model_data.S
    ${CMAKE_CURRENT_BINARY_DIR}/model_data_neo.S
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizer_data_neo.S
)
pico_generate_pio_header(picotales_voice_story ${CMAKE_CURRENT_LIST_DIR}/inmp441.pio)
# .incbin dependency is invisible to CMake — without this, swapping the
# recognizer model silently ships the stale one
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/tinysr_model_data.S PROPERTIES
    OBJECT_DEPENDS ${TINYSR_MODEL_ABS})
target_include_directories(picotales_voice_story PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../tinysr)
target_compile_options(picotales_voice_story PRIVATE -Wno-unused-function)
target_compile_definitions(picotales_voice_story PRIVATE
    PICO_ON_DEVICE=1
    PICO
    NO_MAIN
    NEO_SEQ_CAP=88
    PICO_STACK_SIZE=0x1000
    # malloc must return NULL under memory pressure (default panics) so the
    # hardened TinySR paths can drop frames instead of freezing the machine
    PICO_MALLOC_PANIC=0
)
target_link_libraries(picotales_voice_story
    pico_stdlib hardware_pio hardware_vreg hardware_clocks hardware_i2c
)
pico_enable_stdio_usb(picotales_voice_story 1)
pico_enable_stdio_uart(picotales_voice_story 0)
pico_add_extra_outputs(picotales_voice_story)
add_custom_command(TARGET picotales_voice_story POST_BUILD
    COMMAND arm-none-eabi-size $<TARGET_FILE:picotales_voice_story>
    COMMENT "Voice story firmware sizes:"
)
endif()
