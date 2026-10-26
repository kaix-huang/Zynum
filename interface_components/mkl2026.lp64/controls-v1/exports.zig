// Generated controls forward to the existing LAPACKE binding configuration.
const binding = @import("lapacke-interface-owner");
pub const BindingOwner = binding;
pub const OwnerIdentity = binding.Provider.OwnerIdentity;
pub const WireInt = c_int;
const ControlInt = c_int;

fn setNancheck(flag: ControlInt) callconv(.c) void {
    binding.shared_services.setNancheck(flag);
}
fn getNancheck() callconv(.c) ControlInt {
    return binding.shared_services.getNancheck();
}
comptime {
    if (@sizeOf(ControlInt) != 4 or @alignOf(ControlInt) != 4) @compileError("LAPACKE controls require the reviewed C int ABI");
    @export(&setNancheck, .{ .name = "LAPACKE_set_nancheck" });
    @export(&getNancheck, .{ .name = "LAPACKE_get_nancheck" });
}
