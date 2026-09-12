if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE precopy_mamba_align_v310_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME PrecopyMambaAlignV310
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE precopy_mamba_align_v310 ACLNNTYPE aclnn)
endif()
