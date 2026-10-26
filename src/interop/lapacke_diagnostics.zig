// Copyright (C) 2026 Zynum contributors
// SPDX-License-Identifier: LGPL-3.0-or-later

//! Interface-only translation of internal diagnostics. No global callback,
//! execution state, allocator or numerical implementation is owned here.
//! The selected export profile supplies both handlers; LAPACKE and underlying
//! Fortran errors deliberately retain their different names and INFO signs.
const std = @import("std");
const common = @import("lapacke_common.zig");

pub fn Diagnostics(comptime Int: type) type {
    if (Int != i32 and Int != i64) @compileError("LAPACKE diagnostics require signed LP64 or ILP64 integers");
    return struct {
        pub const Channel = enum { lapacke, fortran };
        pub const Message = struct {
            channel: Channel,
            storage: [20:0]u8 = @splat(0),
            length: usize,
            info: Int,

            // The caller retains this value for the entire synchronous callback.
            pub fn name(self: *const Message) [:0]const u8 {
                return self.storage[0..self.length :0];
            }
        };

        pub fn translate(event: common.ErrorEvent) error{InvalidEvent}!Message {
            if (std.mem.indexOfScalar(u8, "sdcz", event.scalar) == null) return error.InvalidEvent;
            const routine = @tagName(event.routine);
            var message: Message = .{ .channel = undefined, .length = 0, .info = event.argument };
            switch (event.kind) {
                .lapacke => {
                    if (event.entry != .high and event.entry != .work) return error.InvalidEvent;
                    if (event.argument >= 0) return error.InvalidEvent;
                    message.channel = .lapacke;
                    @memcpy(message.storage[0..8], "LAPACKE_");
                    message.storage[8] = event.scalar;
                    @memcpy(message.storage[9..][0..routine.len], routine);
                    message.length = 9 + routine.len;
                    if (event.entry == .work) {
                        @memcpy(message.storage[message.length..][0..5], "_work");
                        message.length += 5;
                    }
                },
                .fortran_returning_test => {
                    const expected: common.Entry = switch (event.routine) {
                        .potrf => .fortran_potrf,
                        .potrs => .fortran_potrs,
                        .getrf => .fortran_getrf,
                        .getrs => .fortran_getrs,
                        .gesv => .fortran_gesv,
                        .posv => .fortran_posv,
                        .trtrs => .fortran_trtrs,
                        .trtri => .fortran_trtri,
                        .getri => .fortran_getri,
                        .potri => .fortran_potri,
                        .stev => .fortran_stev,
                        .gelsd => .fortran_gelsd,
                        .gels => .fortran_gels,
                        .gelss => .fortran_gelss,
                        .lalsd => .fortran_lalsd,
                        .gesvd => .fortran_gesvd,
                        .heev => .fortran_heev,
                        .syev => .fortran_syev,
                        .sygv => .fortran_sygv,
                        .syevd => .fortran_syevd,
                        .heevd => .fortran_heevd,
                        .sygvd => .fortran_sygvd,
                        .hegvd => .fortran_hegvd,
                        .hegv => .fortran_hegv,
                        .geqrf => .fortran_geqrf,
                        .ormqr => .fortran_ormqr,
                        .unmqr => .fortran_unmqr,
                        .orgqr => .fortran_orgqr,
                        .ungqr => .fortran_ungqr,
                    };
                    if (event.entry != expected or event.argument <= 0) return error.InvalidEvent;
                    message.channel = .fortran;
                    message.storage[0] = std.ascii.toUpper(event.scalar);
                    for (routine, 1..) |byte, index| message.storage[index] = std.ascii.toUpper(byte);
                    message.length = 1 + routine.len;
                },
            }
            return message;
        }

        /// Handler signatures are internal contracts, not claims about a
        /// vendor's Fortran hidden-length ABI or dynamic handler replacement.
        /// A profile bridges these names/integers to its precise external ABI.
        pub fn dispatch(event: common.ErrorEvent, handlers: anytype) error{InvalidEvent}!void {
            const message = try translate(event);
            switch (message.channel) {
                .lapacke => handlers.lapacke(message.name(), message.info),
                .fortran => handlers.fortran(message.name(), message.info),
            }
        }
    };
}

test "diagnostics preserve routine, scalar, entry, channel and integer model" {
    inline for (.{ i32, i64 }) |Int| {
        const D = Diagnostics(Int);
        inline for ([_]common.Routine{ .potrf, .potrs, .getrf, .getrs, .gesv, .gesvd }) |routine| {
            for ("sdcz") |scalar| {
                inline for (.{ common.Entry.high, .work }) |entry| {
                    const message = try D.translate(.{ .kind = .lapacke, .entry = entry, .scalar = scalar, .argument = -5, .routine = routine });
                    var buffer: [32]u8 = undefined;
                    const expected = try std.fmt.bufPrint(&buffer, "LAPACKE_{c}{s}{s}", .{ scalar, @tagName(routine), if (entry == .work) "_work" else "" });
                    try std.testing.expectEqualStrings(expected, message.name());
                    try std.testing.expectEqual(D.Channel.lapacke, message.channel);
                    try std.testing.expectEqual(@as(Int, -5), message.info);
                    try std.testing.expectEqual(@as(u8, 0), message.name()[message.length]);
                }
                const entry: common.Entry = switch (routine) {
                    .potrf => .fortran_potrf,
                    .potrs => .fortran_potrs,
                    .getrf => .fortran_getrf,
                    .getrs => .fortran_getrs,
                    .gesv => .fortran_gesv,
                    .posv => .fortran_posv,
                    .trtrs => .fortran_trtrs,
                    .trtri => .fortran_trtri,
                    .getri => .fortran_getri,
                    .potri => .fortran_potri,
                    .stev => .fortran_stev,
                    .gelsd => .fortran_gelsd,
                    .gels => .fortran_gels,
                    .gelss => .fortran_gelss,
                    .gesvd => .fortran_gesvd,
                    .lalsd => .fortran_lalsd,
                    .heev => .fortran_heev,
                    .syev => .fortran_syev,
                    .sygv => .fortran_sygv,
                    .syevd => .fortran_syevd,
                    .heevd => .fortran_heevd,
                    .sygvd => .fortran_sygvd,
                    .hegvd => .fortran_hegvd,
                    .hegv => .fortran_hegv,
                    .geqrf => .fortran_geqrf,
                    .ormqr => .fortran_ormqr,
                    .unmqr => .fortran_unmqr,
                    .orgqr => .fortran_orgqr,
                    .ungqr => .fortran_ungqr,
                };
                const message = try D.translate(.{ .kind = .fortran_returning_test, .entry = entry, .scalar = scalar, .argument = 4, .routine = routine });
                var expected: [6]u8 = undefined;
                expected[0] = std.ascii.toUpper(scalar);
                for (@tagName(routine), 1..) |byte, index| expected[index] = std.ascii.toUpper(byte);
                try std.testing.expectEqualStrings(expected[0 .. 1 + @tagName(routine).len], message.name());
                try std.testing.expectEqual(D.Channel.fortran, message.channel);
                try std.testing.expectEqual(@as(Int, 4), message.info);
            }
        }
        for ([_]i32{ -1010, -1011, std.math.minInt(i32) }) |info| {
            const message = try D.translate(.{ .kind = .lapacke, .entry = .work, .scalar = 'd', .argument = info });
            try std.testing.expectEqual(@as(Int, info), message.info);
        }
    }
}

test "invalid diagnostic events never call either handler" {
    const RejectHandlers = struct {
        fn lapacke(_: @This(), _: [:0]const u8, _: i32) void {
            @panic("invalid LAPACKE callback");
        }
        fn fortran(_: @This(), _: [:0]const u8, _: i32) void {
            @panic("invalid Fortran callback");
        }
    };
    const D = Diagnostics(i32);
    for ([_]common.ErrorEvent{
        .{ .kind = .lapacke, .entry = .high, .scalar = 'q', .argument = -1 },
        .{ .kind = .lapacke, .entry = .high, .scalar = 'd', .argument = 0 },
        .{ .kind = .lapacke, .entry = .fortran_potrf, .scalar = 'd', .argument = -1 },
        .{ .kind = .fortran_returning_test, .entry = .fortran_potrs, .scalar = 'd', .argument = 1 },
        .{ .kind = .fortran_returning_test, .entry = .fortran_potrf, .scalar = 'd', .argument = -1 },
    }) |event| try std.testing.expectError(error.InvalidEvent, D.dispatch(event, RejectHandlers{}));
}

test "valid diagnostics dispatch once to the selected handler" {
    inline for (.{ i32, i64 }) |Int| {
        const Recorder = struct {
            lapacke_count: usize = 0,
            fortran_count: usize = 0,
            last_info: Int = 0,
            last_name: [20]u8 = @splat(0),
            length: usize = 0,
            fn record(self: *@This(), name: [:0]const u8, info: Int) void {
                self.length = name.len;
                @memcpy(self.last_name[0..name.len], name);
                self.last_info = info;
            }
            fn lapacke(self: *@This(), name: [:0]const u8, info: Int) void {
                self.lapacke_count += 1;
                self.record(name, info);
            }
            fn fortran(self: *@This(), name: [:0]const u8, info: Int) void {
                self.fortran_count += 1;
                self.record(name, info);
            }
        };
        var recorder: Recorder = .{};
        const D = Diagnostics(Int);
        try D.dispatch(.{ .kind = .lapacke, .entry = .work, .scalar = 'z', .argument = -1011, .routine = .potrf }, &recorder);
        try std.testing.expectEqual(@as(usize, 1), recorder.lapacke_count);
        try std.testing.expectEqual(@as(usize, 0), recorder.fortran_count);
        try std.testing.expectEqualStrings("LAPACKE_zpotrf_work", recorder.last_name[0..recorder.length]);
        try std.testing.expectEqual(@as(Int, -1011), recorder.last_info);
        try D.dispatch(.{ .kind = .fortran_returning_test, .entry = .fortran_getrs, .scalar = 'd', .argument = 2, .routine = .getrs }, &recorder);
        try std.testing.expectEqual(@as(usize, 1), recorder.lapacke_count);
        try std.testing.expectEqual(@as(usize, 1), recorder.fortran_count);
        try std.testing.expectEqualStrings("DGETRS", recorder.last_name[0..recorder.length]);
        try std.testing.expectEqual(@as(Int, 2), recorder.last_info);
    }
}
