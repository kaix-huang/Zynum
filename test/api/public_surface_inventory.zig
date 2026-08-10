// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Declarative baseline for the reachable public Zig BLAS surface.
//!
//! Source paths are repository-relative declaration owners. A namespace is
//! listed once per import surface even when several surfaces reach the same
//! source file.

const std = @import("std");

pub const Surface = enum {
    zynum,
    zynum_blas,
    zynum_dot_blas,
};

pub const NamespaceRole = enum {
    root,
    types,
    runtime,
    api,
    api_views,
    api_aliasing,
    api_operations,
};

pub const NamespaceId = enum {
    zynum,
    zynum_types,
    zynum_runtime,
    zynum_api,
    zynum_api_views,
    zynum_api_aliasing,
    zynum_api_operations,
    zynum_blas,
    zynum_blas_types,
    zynum_blas_runtime,
    zynum_blas_api,
    zynum_blas_api_views,
    zynum_blas_api_aliasing,
    zynum_blas_api_operations,
    zynum_dot_blas,
    zynum_dot_blas_types,
    zynum_dot_blas_runtime,
    zynum_dot_blas_api,
    zynum_dot_blas_api_views,
    zynum_dot_blas_api_aliasing,
    zynum_dot_blas_api_operations,
};

pub const ForwardingClass = enum {
    declaration_owner,
    compatibility_facade,
    module_facade,
    namespace_alias,
    transitive_namespace_alias,
    api_facade,
    direct_namespace_import,
};

pub const NominalModuleInstance = enum {
    top_level_package,
    standalone_blas_module,
};

pub const Namespace = struct {
    id: NamespaceId,
    surface: Surface,
    role: NamespaceRole,
    namespace_path: []const u8,
    owner_namespace: []const u8,
    source_path: []const u8,
    nominal_module_instance: NominalModuleInstance,
    forwarding_path: []const u8,
    forwarding_class: ForwardingClass,
    public_declarations: []const []const u8,
};

pub const ClosureKind = enum {
    return_type,
    error_union_payload,
    parameter_type,
    public_absence,
};

pub const Visibility = enum {
    public,
    internal_public,
    lexical_private,
    absent,
};

pub const SignatureClosure = struct {
    surface: Surface,
    declaration_path: []const u8,
    kind: ClosureKind,
    signature_fragment: []const u8,
    type_owner_namespace: []const u8,
    source_path: []const u8,
    forwarding_path: []const u8,
    visibility: Visibility,
};

pub const DeclarationKind = enum {
    namespace,
    type_alias,
    nominal_type,
    error_set,
    constant,
    function,
    generic_function,
};

pub const OwnerDeclaration = struct {
    owner_namespace: []const u8,
    declaration_name: []const u8,
    source_path: []const u8,
    kind: DeclarationKind,
    raw_signature: []const u8,
    semantics: []const u8,
};

pub const FacadeForwardingClass = enum {
    namespace_alias,
    type_alias,
    function_alias,
};

pub const FacadeForwarding = struct {
    surface: Surface,
    namespace_path: []const u8,
    declaration_name: []const u8,
    canonical_owner_namespace: []const u8,
    canonical_declaration_name: []const u8,
    class: FacadeForwardingClass,
};

pub const ModuleInstanceRelation = struct {
    left_surface: Surface,
    right_surface: Surface,
    coimport_supported: bool,
    nominal_types_equal: bool,
    nominal_signature_types_equal: bool,
    recorded_type_names_equal: bool,
    error_sets_structurally_equal: bool,
    relation: []const u8,
};

pub const root_declarations: []const []const u8 = &.{
    "types",
    "runtime",
    "api",
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "BlasError",
    "Error",
    "MatrixTransform",
    "MatrixOperation",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiplyWorkspaceLength",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const zynum_root_declarations: []const []const u8 = &.{
    "blas",
    "types",
    "runtime",
    "api",
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "BlasError",
    "Error",
    "MatrixTransform",
    "MatrixOperation",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiplyWorkspaceLength",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const zynum_blas_root_declarations: []const []const u8 = &.{
    "types",
    "runtime",
    "api",
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "BlasError",
    "Error",
    "MatrixTransform",
    "MatrixOperation",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiplyWorkspaceLength",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const zynum_dot_blas_root_declarations: []const []const u8 = &.{
    "types",
    "runtime",
    "api",
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "BlasError",
    "Error",
    "MatrixTransform",
    "MatrixOperation",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiplyWorkspaceLength",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const types_declarations: []const []const u8 = &.{
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "Layout",
    "Transpose",
    "Uplo",
    "Diag",
    "Side",
    "complexF32",
    "complexF64",
};

pub const runtime_declarations: []const []const u8 = &.{
    "maximum_threads_env_name",
    "worker_stack_size",
    "setMaxThreads",
    "maxThreadsOverride",
    "totalThreadCount",
    "maxThreads",
    "helperThreadCount",
    "hasExplicitThreadLimit",
    "performanceThreadCount",
    "efficiencyThreadCount",
    "performanceL2Bytes",
    "cacheLineBytes",
    "configureWorkerThread",
};

pub const api_declarations: []const []const u8 = &.{
    "views",
    "aliasing",
    "operations",
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "Error",
    "BlasError",
    "MatrixTransform",
    "MatrixOperation",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiplyWorkspaceLength",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const views_declarations: []const []const u8 = &.{
    "BlasInt",
    "ComplexF32",
    "ComplexF64",
    "Error",
    "BlasError",
    "runtime_checks_enabled",
    "MatrixTransform",
    "MatrixOperation",
    "expectScalarType",
    "toCoreTranspose",
    "requiredVectorStorageLength",
    "requiredMatrixStorageLength",
    "validateVectorStorage",
    "validateMatrixStorage",
    "optionField",
    "ConstVector",
    "Vector",
    "ConstMatrix",
    "Matrix",
    "constVector",
    "vector",
    "constMatrix",
    "matrix",
};

pub const aliasing_declarations: []const []const u8 = &.{
    "vectorsExactlyMatch",
    "vectorRange",
    "matrixRange",
    "vectorsOverlap",
    "vectorMatrixOverlap",
    "matricesOverlap",
    "ensureNoVectorOverlap",
    "ensureNoVectorMatrixOverlap",
    "ensureNoMatrixOverlap",
    "ensureNoPartialVectorOverlap",
};

pub const operations_declarations: []const []const u8 = &.{
    "Error",
    "BlasError",
    "matrixVectorMultiplyWorkspaceLength",
    "matrixMultiplyWorkspaceLength",
    "swapVectors",
    "copyVector",
    "scaleVector",
    "scaleVectorInto",
    "addScaledVector",
    "addScaledVectorInto",
    "combineVectors",
    "combineVectorsInto",
    "dotProduct",
    "conjugatedDotProduct",
    "euclideanNorm",
    "matrixVectorMultiply",
    "matrixVectorMultiplyWithWorkspace",
    "matrixMultiply",
    "matrixMultiplyWithWorkspace",
};

pub const namespaces = [_]Namespace{
    namespace(.zynum, .zynum, .root, "zynum", "zynum", "src/zynum.zig", "zynum -> zynum.blas", .compatibility_facade, zynum_root_declarations),
    namespace(.zynum_types, .zynum, .types, "zynum.types", "blas.types", "src/blas/types.zig", "zynum.types -> zynum.blas.types", .namespace_alias, types_declarations),
    namespace(.zynum_runtime, .zynum, .runtime, "zynum.runtime", "blas.runtime", "src/blas/runtime.zig", "zynum.runtime -> zynum.blas.runtime", .namespace_alias, runtime_declarations),
    namespace(.zynum_api, .zynum, .api, "zynum.api", "blas.api", "src/blas/api.zig", "zynum.api -> zynum.blas.api", .namespace_alias, api_declarations),
    namespace(.zynum_api_views, .zynum, .api_views, "zynum.api.views", "blas.api.views", "src/blas/api/views.zig", "zynum.api.views -> zynum.blas.api.views", .transitive_namespace_alias, views_declarations),
    namespace(.zynum_api_aliasing, .zynum, .api_aliasing, "zynum.api.aliasing", "blas.api.aliasing", "src/blas/api/aliasing.zig", "zynum.api.aliasing -> zynum.blas.api.aliasing", .transitive_namespace_alias, aliasing_declarations),
    namespace(.zynum_api_operations, .zynum, .api_operations, "zynum.api.operations", "blas.api.operations", "src/blas/api/operations.zig", "zynum.api.operations -> zynum.blas.api.operations", .transitive_namespace_alias, operations_declarations),

    namespace(.zynum_blas, .zynum_blas, .root, "zynum-blas", "blas", "src/blas.zig", "zynum-blas -> types + runtime + api", .module_facade, zynum_blas_root_declarations),
    namespace(.zynum_blas_types, .zynum_blas, .types, "zynum-blas.types", "blas.types", "src/blas/types.zig", "zynum-blas.types -> src/blas/types.zig", .direct_namespace_import, types_declarations),
    namespace(.zynum_blas_runtime, .zynum_blas, .runtime, "zynum-blas.runtime", "blas.runtime", "src/blas/runtime.zig", "zynum-blas.runtime -> src/blas/runtime.zig", .direct_namespace_import, runtime_declarations),
    namespace(.zynum_blas_api, .zynum_blas, .api, "zynum-blas.api", "blas.api", "src/blas/api.zig", "zynum-blas.api -> src/blas/api.zig", .api_facade, api_declarations),
    namespace(.zynum_blas_api_views, .zynum_blas, .api_views, "zynum-blas.api.views", "blas.api.views", "src/blas/api/views.zig", "zynum-blas.api.views -> src/blas/api/views.zig", .direct_namespace_import, views_declarations),
    namespace(.zynum_blas_api_aliasing, .zynum_blas, .api_aliasing, "zynum-blas.api.aliasing", "blas.api.aliasing", "src/blas/api/aliasing.zig", "zynum-blas.api.aliasing -> src/blas/api/aliasing.zig", .direct_namespace_import, aliasing_declarations),
    namespace(.zynum_blas_api_operations, .zynum_blas, .api_operations, "zynum-blas.api.operations", "blas.api.operations", "src/blas/api/operations.zig", "zynum-blas.api.operations -> src/blas/api/operations.zig", .direct_namespace_import, operations_declarations),

    namespace(.zynum_dot_blas, .zynum_dot_blas, .root, "zynum.blas", "blas", "src/blas.zig", "zynum.blas -> types + runtime + api", .module_facade, zynum_dot_blas_root_declarations),
    namespace(.zynum_dot_blas_types, .zynum_dot_blas, .types, "zynum.blas.types", "blas.types", "src/blas/types.zig", "zynum.blas.types -> src/blas/types.zig", .direct_namespace_import, types_declarations),
    namespace(.zynum_dot_blas_runtime, .zynum_dot_blas, .runtime, "zynum.blas.runtime", "blas.runtime", "src/blas/runtime.zig", "zynum.blas.runtime -> src/blas/runtime.zig", .direct_namespace_import, runtime_declarations),
    namespace(.zynum_dot_blas_api, .zynum_dot_blas, .api, "zynum.blas.api", "blas.api", "src/blas/api.zig", "zynum.blas.api -> src/blas/api.zig", .api_facade, api_declarations),
    namespace(.zynum_dot_blas_api_views, .zynum_dot_blas, .api_views, "zynum.blas.api.views", "blas.api.views", "src/blas/api/views.zig", "zynum.blas.api.views -> src/blas/api/views.zig", .direct_namespace_import, views_declarations),
    namespace(.zynum_dot_blas_api_aliasing, .zynum_dot_blas, .api_aliasing, "zynum.blas.api.aliasing", "blas.api.aliasing", "src/blas/api/aliasing.zig", "zynum.blas.api.aliasing -> src/blas/api/aliasing.zig", .direct_namespace_import, aliasing_declarations),
    namespace(.zynum_dot_blas_api_operations, .zynum_dot_blas, .api_operations, "zynum.blas.api.operations", "blas.api.operations", "src/blas/api/operations.zig", "zynum.blas.api.operations -> src/blas/api/operations.zig", .direct_namespace_import, operations_declarations),
};

pub const signature_closure = [_]SignatureClosure{
    closure(.zynum, "zynum.api.views.toCoreTranspose", .return_type, "fn (MatrixTransform) TransposeMode", "blas.core.shared.scalar.TransposeMode", "src/blas/core/shared/scalar.zig", "zynum.api.views -> zynum.blas.api.views -> blas.core.TransposeMode", .internal_public),
    closure(.zynum_blas, "zynum-blas.api.views.toCoreTranspose", .return_type, "fn (MatrixTransform) TransposeMode", "blas.core.shared.scalar.TransposeMode", "src/blas/core/shared/scalar.zig", "zynum-blas.api.views -> blas.core.TransposeMode", .internal_public),
    closure(.zynum_dot_blas, "zynum.blas.api.views.toCoreTranspose", .return_type, "fn (MatrixTransform) TransposeMode", "blas.core.shared.scalar.TransposeMode", "src/blas/core/shared/scalar.zig", "zynum.blas.api.views -> blas.core.TransposeMode", .internal_public),

    closure(.zynum, "zynum.api.aliasing.vectorRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum.api.aliasing -> zynum.blas.api.aliasing.ByteRange", .lexical_private),
    closure(.zynum, "zynum.api.aliasing.matrixRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum.api.aliasing -> zynum.blas.api.aliasing.ByteRange", .lexical_private),
    closure(.zynum_blas, "zynum-blas.api.aliasing.vectorRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum-blas.api.aliasing -> blas.api.aliasing.ByteRange", .lexical_private),
    closure(.zynum_blas, "zynum-blas.api.aliasing.matrixRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum-blas.api.aliasing -> blas.api.aliasing.ByteRange", .lexical_private),
    closure(.zynum_dot_blas, "zynum.blas.api.aliasing.vectorRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum.blas.api.aliasing -> blas.api.aliasing.ByteRange", .lexical_private),
    closure(.zynum_dot_blas, "zynum.blas.api.aliasing.matrixRange", .error_union_payload, "fn (type, anytype) Error!ByteRange", "blas.api.aliasing.ByteRange", "src/blas/api/aliasing.zig", "zynum.blas.api.aliasing -> blas.api.aliasing.ByteRange", .lexical_private),

    closure(.zynum, "zynum.runtime.configureWorkerThread", .parameter_type, "fn (?usize) void", "blas.runtime", "src/blas/runtime.zig", "zynum.runtime -> zynum.blas.runtime.configureWorkerThread", .public),
    closure(.zynum_blas, "zynum-blas.runtime.configureWorkerThread", .parameter_type, "fn (?usize) void", "blas.runtime", "src/blas/runtime.zig", "zynum-blas.runtime.configureWorkerThread", .public),
    closure(.zynum_dot_blas, "zynum.blas.runtime.configureWorkerThread", .parameter_type, "fn (?usize) void", "blas.runtime", "src/blas/runtime.zig", "zynum.blas.runtime.configureWorkerThread", .public),

    closure(.zynum, "zynum.runtime.shutdown", .public_absence, "absent", "none", "src/blas/runtime.zig", "no public Zig forwarding path", .absent),
    closure(.zynum_blas, "zynum-blas.runtime.shutdown", .public_absence, "absent", "none", "src/blas/runtime.zig", "no public Zig forwarding path", .absent),
    closure(.zynum_dot_blas, "zynum.blas.runtime.shutdown", .public_absence, "absent", "none", "src/blas/runtime.zig", "no public Zig forwarding path", .absent),
};

/// One explicit row per declaration owned by the public leaf namespaces.
pub const owner_declarations = [_]OwnerDeclaration{
    owner("blas.types", "BlasInt", "src/blas/types.zig", .type_alias, "const BlasInt = i32", "signed 32-bit BLAS integer"),
    owner("blas.types", "ComplexF32", "src/blas/types.zig", .nominal_type, "extern struct { re: f32, im: f32 }", "two-component complex f32 ABI value"),
    owner("blas.types", "ComplexF64", "src/blas/types.zig", .nominal_type, "extern struct { re: f64, im: f64 }", "two-component complex f64 ABI value"),
    owner("blas.types", "Layout", "src/blas/types.zig", .nominal_type, "enum(c_int) { row_major = 101, col_major = 102 }", "matrix storage layout code"),
    owner("blas.types", "Transpose", "src/blas/types.zig", .nominal_type, "enum(c_int) { no_trans = 111, trans = 112, conj_trans = 113 }", "matrix transpose code"),
    owner("blas.types", "Uplo", "src/blas/types.zig", .nominal_type, "enum(c_int) { upper = 121, lower = 122 }", "stored triangle code"),
    owner("blas.types", "Diag", "src/blas/types.zig", .nominal_type, "enum(c_int) { non_unit = 131, unit = 132 }", "diagonal interpretation code"),
    owner("blas.types", "Side", "src/blas/types.zig", .nominal_type, "enum(c_int) { left = 141, right = 142 }", "matrix operand side code"),
    owner("blas.types", "complexF32", "src/blas/types.zig", .function, "fn (re: f32, im: f32) ComplexF32", "construct ComplexF32 with result.re = re and result.im = im"),
    owner("blas.types", "complexF64", "src/blas/types.zig", .function, "fn (re: f64, im: f64) ComplexF64", "construct ComplexF64 with result.re = re and result.im = im"),

    owner("blas.runtime", "maximum_threads_env_name", "src/blas/runtime.zig", .constant, "const maximum_threads_env_name = \"ZYNUM_MAXIMUM_THREADS\"", "supported process thread-limit name"),
    owner("blas.runtime", "worker_stack_size", "src/blas/runtime.zig", .constant, "usize = 2 * 1024 * 1024", "worker stack byte count"),
    owner("blas.runtime", "setMaxThreads", "src/blas/runtime.zig", .function, "fn (n: usize) void", "set the process-local thread override"),
    owner("blas.runtime", "maxThreadsOverride", "src/blas/runtime.zig", .function, "fn () usize", "read the process-local thread override"),
    owner("blas.runtime", "totalThreadCount", "src/blas/runtime.zig", .function, "fn () usize", "return detected total logical threads"),
    owner("blas.runtime", "maxThreads", "src/blas/runtime.zig", .function, "fn () usize", "return the effective capped thread count"),
    owner("blas.runtime", "helperThreadCount", "src/blas/runtime.zig", .function, "fn (max_helpers: usize) usize", "cap helper threads below the effective total"),
    owner("blas.runtime", "hasExplicitThreadLimit", "src/blas/runtime.zig", .function, "fn () bool", "report whether an override or supported process limit is active"),
    owner("blas.runtime", "performanceThreadCount", "src/blas/runtime.zig", .function, "fn () usize", "return detected performance-thread capacity or zero"),
    owner("blas.runtime", "efficiencyThreadCount", "src/blas/runtime.zig", .function, "fn () usize", "return detected efficiency-thread capacity or zero"),
    owner("blas.runtime", "performanceL2Bytes", "src/blas/runtime.zig", .function, "fn () usize", "return performance-cluster L2 capacity or fallback"),
    owner("blas.runtime", "cacheLineBytes", "src/blas/runtime.zig", .function, "fn () usize", "return detected cache-line bytes or fallback"),
    owner("blas.runtime", "configureWorkerThread", "src/blas/runtime.zig", .function, "fn (affinity_ordinal: ?usize) void", "apply supported worker QoS and optional affinity"),

    owner("blas.api.views", "BlasInt", "src/blas/api/views.zig", .type_alias, "const BlasInt = types.BlasInt", "public view dimension and stride integer"),
    owner("blas.api.views", "ComplexF32", "src/blas/api/views.zig", .type_alias, "const ComplexF32 = types.ComplexF32", "supported complex f32 scalar"),
    owner("blas.api.views", "ComplexF64", "src/blas/api/views.zig", .type_alias, "const ComplexF64 = types.ComplexF64", "supported complex f64 scalar"),
    owner("blas.api.views", "Error", "src/blas/api/views.zig", .error_set, "error{ DimensionMismatch, InvalidLength, InvalidStride, InvalidLeadingDimension, BufferTooSmall, WorkspaceTooSmall, AliasingNotAllowed }", "exact structural checked Zig API error set"),
    owner("blas.api.views", "BlasError", "src/blas/api/views.zig", .type_alias, "const BlasError = Error", "backwards-compatible error alias"),
    owner("blas.api.views", "runtime_checks_enabled", "src/blas/api/views.zig", .constant, "const runtime_checks_enabled = builtin.mode != .ReleaseFast", "enable capacity and alias checks outside ReleaseFast"),
    owner("blas.api.views", "MatrixTransform", "src/blas/api/views.zig", .nominal_type, "enum { normal, transposed, adjoint }", "logical matrix transform"),
    owner("blas.api.views", "MatrixOperation", "src/blas/api/views.zig", .type_alias, "const MatrixOperation = MatrixTransform", "backwards-compatible transform alias"),
    owner("blas.api.views", "expectScalarType", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type) void", "accept only f32, f64, ComplexF32, or ComplexF64"),
    owner("blas.api.views", "toCoreTranspose", "src/blas/api/views.zig", .function, "fn (transform: MatrixTransform) core.TransposeMode", "map the public transform to the internal transpose mode"),
    owner("blas.api.views", "requiredVectorStorageLength", "src/blas/api/views.zig", .function, "fn (length: BlasInt, stride: BlasInt) Error!usize", "checked strided-vector storage extent"),
    owner("blas.api.views", "requiredMatrixStorageLength", "src/blas/api/views.zig", .function, "fn (row_count: BlasInt, column_count: BlasInt, leading_dimension: BlasInt) Error!usize", "checked column-major matrix storage extent"),
    owner("blas.api.views", "validateVectorStorage", "src/blas/api/views.zig", .function, "fn (data_len: usize, length: BlasInt, stride: BlasInt) Error!void", "validate vector structure and checked-build capacity"),
    owner("blas.api.views", "validateMatrixStorage", "src/blas/api/views.zig", .function, "fn (data_len: usize, row_count: BlasInt, column_count: BlasInt, leading_dimension: BlasInt) Error!void", "validate matrix structure and checked-build capacity"),
    owner("blas.api.views", "optionField", "src/blas/api/views.zig", .generic_function, "fn (options: anytype, comptime name: []const u8, fallback: anytype) @TypeOf(fallback)", "options and fallback are runtime generic values; name is comptime; return options.name when present, otherwise fallback, with result type exactly @TypeOf(fallback)"),
    owner("blas.api.views", "ConstVector", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type) type", "comptime scalar factory for { values: []const T, length: BlasInt, stride: BlasInt = 1 }; check validates structure in every mode and capacity outside ReleaseFast"),
    owner("blas.api.views", "Vector", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type) type", "comptime scalar factory for { values: []T, length: BlasInt, stride: BlasInt = 1 }; asConst preserves storage, length, and stride; check follows ConstVector validation"),
    owner("blas.api.views", "ConstMatrix", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type) type", "comptime scalar factory for column-major { values, row_count, column_count, leading_dimension, operation = .normal }; transposed/adjoint preserve storage shape and set operation; effective counts swap for either non-normal operation"),
    owner("blas.api.views", "Matrix", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type) type", "comptime scalar factory for mutable column-major { values, row_count, column_count, leading_dimension }; asConst preserves fields and sets operation to .normal"),
    owner("blas.api.views", "constVector", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type, values: []const T, options: anytype) Error!ConstVector(T)", "options is a runtime generic value; length defaults to values.len converted to BlasInt, stride defaults to 1; explicit length/stride are forwarded exactly; validates before returning the unchanged slice"),
    owner("blas.api.views", "vector", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type, values: []T, options: anytype) Error!Vector(T)", "options is a runtime generic value; length defaults to values.len converted to BlasInt, stride defaults to 1; explicit length/stride are forwarded exactly; validates before returning the unchanged slice"),
    owner("blas.api.views", "constMatrix", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type, values: []const T, options: anytype) Error!ConstMatrix(T)", "options is a runtime generic value requiring row_count and column_count; leading_dimension defaults to row_count and explicit leading_dimension is forwarded; operation is always the ConstMatrix default .normal; validates before returning the unchanged slice"),
    owner("blas.api.views", "matrix", "src/blas/api/views.zig", .generic_function, "fn (comptime T: type, values: []T, options: anytype) Error!Matrix(T)", "options is a runtime generic value requiring row_count and column_count; leading_dimension defaults to row_count and explicit leading_dimension is forwarded; validates before returning the unchanged slice"),

    owner("blas.api.aliasing", "vectorsExactlyMatch", "src/blas/api/aliasing.zig", .generic_function, "fn (first: anytype, second: anytype) bool", "both parameters are runtime generic values; true exactly when values.ptr, length, and stride all match"),
    owner("blas.api.aliasing", "vectorRange", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, vector: anytype) views.Error!ByteRange", "T is comptime and vector is runtime generic; returns private ByteRange [values.ptr, values.ptr + requiredVectorStorageLength * @sizeOf(T)) with checked arithmetic"),
    owner("blas.api.aliasing", "matrixRange", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, matrix: anytype) views.Error!ByteRange", "T is comptime and matrix is runtime generic; returns private ByteRange [values.ptr, values.ptr + requiredMatrixStorageLength * @sizeOf(T)) with checked arithmetic"),
    owner("blas.api.aliasing", "vectorsOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, first: anytype, second: anytype) views.Error!bool", "T is comptime and operands are runtime generic; empty ranges never overlap; otherwise uses half-open vector byte ranges"),
    owner("blas.api.aliasing", "vectorMatrixOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, vector: anytype, matrix: anytype) views.Error!bool", "T is comptime and operands are runtime generic; empty ranges never overlap; otherwise compares half-open vector and matrix byte ranges"),
    owner("blas.api.aliasing", "matricesOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, first: anytype, second: anytype) views.Error!bool", "T is comptime and operands are runtime generic; empty ranges never overlap; otherwise uses half-open matrix byte ranges"),
    owner("blas.api.aliasing", "ensureNoVectorOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, first: anytype, second: anytype) views.Error!void", "no-op in ReleaseFast; otherwise returns AliasingNotAllowed exactly when vectorsOverlap is true"),
    owner("blas.api.aliasing", "ensureNoVectorMatrixOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, vector: anytype, matrix: anytype) views.Error!void", "no-op in ReleaseFast; otherwise returns AliasingNotAllowed exactly when vectorMatrixOverlap is true"),
    owner("blas.api.aliasing", "ensureNoMatrixOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, first: anytype, second: anytype) views.Error!void", "no-op in ReleaseFast; otherwise returns AliasingNotAllowed exactly when matricesOverlap is true"),
    owner("blas.api.aliasing", "ensureNoPartialVectorOverlap", "src/blas/api/aliasing.zig", .generic_function, "fn (comptime T: type, first: anytype, second: anytype) views.Error!void", "no-op in ReleaseFast; otherwise permits vectorsExactlyMatch and returns AliasingNotAllowed for any other overlap"),

    owner("blas.api.operations", "Error", "src/blas/api/operations.zig", .type_alias, "const Error = views.Error", "checked operation error alias"),
    owner("blas.api.operations", "BlasError", "src/blas/api/operations.zig", .type_alias, "const BlasError = views.BlasError", "backwards-compatible checked operation error alias"),
    owner("blas.api.operations", "matrixVectorMultiplyWorkspaceLength", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!usize", "arguments is one runtime generic value containing matrix; returns matrix.effectiveRowCount() as usize or InvalidLength when negative"),
    owner("blas.api.operations", "matrixMultiplyWorkspaceLength", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!usize", "arguments is one runtime generic value containing result_matrix; returns checked row_count * column_count or InvalidLength for a negative dimension or overflow"),
    owner("blas.api.operations", "swapVectors", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires first_vector and second_vector with the same Scalar; validates both and swaps min(lengths); exact alias is allowed and partial overlap is checked outside ReleaseFast"),
    owner("blas.api.operations", "copyVector", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires source_vector and destination_vector with the same Scalar; validates both and copies min(lengths); exact alias is a no-op and partial overlap is checked outside ReleaseFast"),
    owner("blas.api.operations", "scaleVector", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires target_vector and scale: Scalar; validates target_vector then performs target_vector *= scale"),
    owner("blas.api.operations", "scaleVectorInto", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires input_vector, result_vector, and scale: Scalar; requires equal lengths; exact input/result alias scales in place and other overlap is checked outside ReleaseFast"),
    owner("blas.api.operations", "addScaledVector", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires source_vector, destination_vector, and scale: Scalar; applies destination += scale * source over min(lengths); exact alias is allowed and partial overlap is checked outside ReleaseFast"),
    owner("blas.api.operations", "addScaledVectorInto", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires equal-length source_vector, input_vector, result_vector, and scale: Scalar; writes result = input + scale * source; result may exactly alias input and other result overlap is checked outside ReleaseFast"),
    owner("blas.api.operations", "combineVectors", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires source_vector, destination_vector, source_scale: Scalar, and destination_scale: Scalar; applies destination = source_scale * source + destination_scale * destination over min(lengths); exact alias is allowed"),
    owner("blas.api.operations", "combineVectorsInto", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires equal-length source_vector, input_vector, result_vector, source_scale: Scalar, and input_scale: Scalar; writes result = source_scale * source + input_scale * input; result may exactly alias input"),
    owner("blas.api.operations", "dotProduct", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!@TypeOf(arguments.left_vector).Scalar", "runtime arguments requires same-Scalar left_vector and right_vector; validates both and returns the unconjugated dot product over min(lengths)"),
    owner("blas.api.operations", "conjugatedDotProduct", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!@TypeOf(arguments.left_vector).Scalar", "runtime arguments requires same-Scalar left_vector and right_vector; validates both and returns the left-conjugated dot product over min(lengths)"),
    owner("blas.api.operations", "euclideanNorm", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!core.Real(@TypeOf(arguments.input_vector).Scalar)", "runtime arguments requires input_vector; validates it and returns core.Real(Scalar), f32 for f32/ComplexF32 and f64 for f64/ComplexF64"),
    owner("blas.api.operations", "matrixVectorMultiply", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires matrix, input_vector, result_vector and optional product_scale/result_scale of Scalar; computes result = product_scale * op(matrix) * input + result_scale * result with defaults product_scale = one and result_scale = zero; validates shapes and rejects result/input or result/matrix overlap outside ReleaseFast"),
    owner("blas.api.operations", "matrixVectorMultiplyWithWorkspace", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments uses the matrixVectorMultiply equation with defaults product_scale = one and result_scale = zero; additionally requires workspace: []Scalar of at least effective result length; permits result/input or result/matrix aliasing but rejects workspace overlap outside ReleaseFast"),
    owner("blas.api.operations", "matrixMultiply", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments requires left_matrix, right_matrix, result_matrix and optional product_scale/result_scale of Scalar; computes result = product_scale * op(left) * op(right) + result_scale * result with defaults product_scale = one and result_scale = zero; validates exact product shape and rejects result/input overlap outside ReleaseFast"),
    owner("blas.api.operations", "matrixMultiplyWithWorkspace", "src/blas/api/operations.zig", .generic_function, "fn (arguments: anytype) Error!void", "runtime arguments uses the matrixMultiply equation with defaults product_scale = one and result_scale = zero; additionally requires workspace: []Scalar of at least result rows * columns; permits result/input aliasing but rejects workspace/input/result overlap outside ReleaseFast"),
};

const zynum_root_forwarding = forwardingRows(.zynum, "zynum", zynum_root_declarations);
const zynum_api_forwarding = forwardingRows(.zynum, "zynum.api", api_declarations);
const zynum_blas_root_forwarding = forwardingRows(.zynum_blas, "zynum-blas", zynum_blas_root_declarations);
const zynum_blas_api_forwarding = forwardingRows(.zynum_blas, "zynum-blas.api", api_declarations);
const zynum_dot_blas_root_forwarding = forwardingRows(.zynum_dot_blas, "zynum.blas", zynum_dot_blas_root_declarations);
const zynum_dot_blas_api_forwarding = forwardingRows(.zynum_dot_blas, "zynum.blas.api", api_declarations);

/// Every facade declaration has an explicit canonical-owner forwarding row.
pub const facade_forwarding = zynum_root_forwarding ++
    zynum_api_forwarding ++
    zynum_blas_root_forwarding ++
    zynum_blas_api_forwarding ++
    zynum_dot_blas_root_forwarding ++
    zynum_dot_blas_api_forwarding;

pub const module_instance_relations = [_]ModuleInstanceRelation{
    .{
        .left_surface = .zynum,
        .right_surface = .zynum_dot_blas,
        .coimport_supported = true,
        .nominal_types_equal = true,
        .nominal_signature_types_equal = true,
        .recorded_type_names_equal = true,
        .error_sets_structurally_equal = true,
        .relation = "zynum.blas is the BLAS namespace in the top-level module instance",
    },
    .{
        .left_surface = .zynum_dot_blas,
        .right_surface = .zynum_blas,
        .coimport_supported = false,
        .nominal_types_equal = false,
        .nominal_signature_types_equal = false,
        .recorded_type_names_equal = true,
        .error_sets_structurally_equal = true,
        .relation = "standalone zynum-blas is separately compiled; the current relative-import graph prevents co-import with zynum.blas",
    },
};

fn owner(
    owner_namespace: []const u8,
    declaration_name: []const u8,
    source_path: []const u8,
    kind: DeclarationKind,
    raw_signature: []const u8,
    semantics: []const u8,
) OwnerDeclaration {
    return .{
        .owner_namespace = owner_namespace,
        .declaration_name = declaration_name,
        .source_path = source_path,
        .kind = kind,
        .raw_signature = raw_signature,
        .semantics = semantics,
    };
}

fn forwardingRows(
    comptime target_surface: Surface,
    comptime namespace_path: []const u8,
    comptime declarations: []const []const u8,
) [declarations.len]FacadeForwarding {
    @setEvalBranchQuota(20_000);
    var rows: [declarations.len]FacadeForwarding = undefined;
    for (declarations, 0..) |name, index| {
        const canonical_owner = canonicalOwner(namespace_path, name);
        rows[index] = .{
            .surface = target_surface,
            .namespace_path = namespace_path,
            .declaration_name = name,
            .canonical_owner_namespace = canonical_owner,
            .canonical_declaration_name = name,
            .class = forwardingClass(name),
        };
    }
    return rows;
}

fn canonicalOwner(comptime namespace_path: []const u8, comptime name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, namespace_path, ".api")) {
        if (std.mem.eql(u8, name, "views") or std.mem.eql(u8, name, "aliasing") or std.mem.eql(u8, name, "operations"))
            return "blas.api";
    } else {
        if (std.mem.eql(u8, name, "blas")) return "blas";
        if (std.mem.eql(u8, name, "types")) return "blas.types";
        if (std.mem.eql(u8, name, "runtime")) return "blas.runtime";
        if (std.mem.eql(u8, name, "api")) return "blas.api";
    }
    if (isOperation(name)) return "blas.api.operations";
    if (std.mem.eql(u8, name, "BlasInt") or std.mem.eql(u8, name, "ComplexF32") or std.mem.eql(u8, name, "ComplexF64"))
        return "blas.types";
    return "blas.api.views";
}

fn forwardingClass(comptime name: []const u8) FacadeForwardingClass {
    if (std.mem.eql(u8, name, "blas") or
        std.mem.eql(u8, name, "types") or
        std.mem.eql(u8, name, "runtime") or
        std.mem.eql(u8, name, "api") or
        std.mem.eql(u8, name, "views") or
        std.mem.eql(u8, name, "aliasing") or
        std.mem.eql(u8, name, "operations")) return .namespace_alias;
    if (std.mem.eql(u8, name, "ConstVector") or
        std.mem.eql(u8, name, "Vector") or
        std.mem.eql(u8, name, "ConstMatrix") or
        std.mem.eql(u8, name, "Matrix")) return .function_alias;
    if (std.ascii.isUpper(name[0])) return .type_alias;
    return .function_alias;
}

fn isOperation(comptime name: []const u8) bool {
    inline for (operations_declarations[2..]) |operation| {
        if (std.mem.eql(u8, name, operation)) return true;
    }
    return false;
}

fn namespace(
    id: NamespaceId,
    surface: Surface,
    role: NamespaceRole,
    namespace_path: []const u8,
    owner_namespace: []const u8,
    source_path: []const u8,
    forwarding_path: []const u8,
    forwarding_class: ForwardingClass,
    public_declarations: []const []const u8,
) Namespace {
    return .{
        .id = id,
        .surface = surface,
        .role = role,
        .namespace_path = namespace_path,
        .owner_namespace = owner_namespace,
        .source_path = source_path,
        .nominal_module_instance = if (surface == .zynum_blas) .standalone_blas_module else .top_level_package,
        .forwarding_path = forwarding_path,
        .forwarding_class = forwarding_class,
        .public_declarations = public_declarations,
    };
}

fn closure(
    surface: Surface,
    declaration_path: []const u8,
    kind: ClosureKind,
    signature_fragment: []const u8,
    type_owner_namespace: []const u8,
    source_path: []const u8,
    forwarding_path: []const u8,
    visibility: Visibility,
) SignatureClosure {
    return .{
        .surface = surface,
        .declaration_path = declaration_path,
        .kind = kind,
        .signature_fragment = signature_fragment,
        .type_owner_namespace = type_owner_namespace,
        .source_path = source_path,
        .forwarding_path = forwarding_path,
        .visibility = visibility,
    };
}
