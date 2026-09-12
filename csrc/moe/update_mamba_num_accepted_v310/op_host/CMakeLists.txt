if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE update_mamba_num_accepted_v310_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME UpdateMambaNumAcceptedV310
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE update_mamba_num_accepted_v310 ACLNNTYPE aclnn)
endif()
