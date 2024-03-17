const std = @import("std");

pub fn readClassificationDataset(n: comptime_int, file_path: []const u8, allocator: std.mem.Allocator) !*[n]bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Opening file {s}\n", .{file_path});
    const file = try std.fs.cwd().openFile(file_path, std.fs.File.OpenFlags{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, n);
    var result: *[n]bool = try allocator.create([n]bool);
    for (0..n) |i| {
        result[i] = contents[i] == @as(u8, 1);
    }
    return result;
}

pub fn readFloatDataset(Precision: type, comptime num_features: comptime_int, comptime num_samples: comptime_int, file_path: []const u8, allocator: std.mem.Allocator) !*[num_samples][num_features]Precision {
    const file = try std.fs.cwd().openFile(file_path, std.fs.File.OpenFlags{});
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    const reader = buf_reader.reader();

    const result: *[num_samples][num_features]Precision = try allocator.create([num_samples][num_features]Precision);
    errdefer allocator.destroy(result);
    for (0..num_samples) |i| {
        const bytes = try reader.readBytesNoEof(num_features);

        for (bytes, 0..) |b, j| {
            result[i][j] = @as(Precision, @floatFromInt(b)) / 255.0;
        }
    }
    return result;
}
