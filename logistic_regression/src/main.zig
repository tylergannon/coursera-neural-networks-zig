const std = @import("std");
const load = @import("./load_data.zig");
const regression = @import("./logistic_regression.zig");

const ATTRIBUTE_COUNT_N = 12288;
const SAMPLE_SIZE = 209;
const TEST_SAMPLE_SIZE = 50;

const Model = regression.model(ATTRIBUTE_COUNT_N, SAMPLE_SIZE);
const TestModel = regression.model(ATTRIBUTE_COUNT_N, TEST_SAMPLE_SIZE);

pub fn main() !void {
    // Prints to stderr (it's a shortcut based on `std.io.getStdErr()`)
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});

    // stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    if (std.os.argv.len >= 4) {
        var gpa = std.heap.GeneralPurposeAllocator(.{
            .thread_safe = false,
            .safety = false,
        }){};

        defer _ = gpa.deinit();
        const allocator = gpa.allocator();
        var args = std.process.args();
        _ = args.skip();
        const train_x_fname = args.next().?;
        const train_y_fname = args.next().?;

        try stdout.print("Argument 0: {s}\n", .{train_x_fname});
        try stdout.print("Argument 1: {s}\n", .{train_y_fname});
        const train_x = try load.readFloatDataset(regression.Precision, ATTRIBUTE_COUNT_N, SAMPLE_SIZE, train_x_fname, allocator);
        const train_y = try load.readClassificationDataset(SAMPLE_SIZE, train_y_fname, allocator);
        const test_x = try load.readFloatDataset(regression.Precision, ATTRIBUTE_COUNT_N, TEST_SAMPLE_SIZE, args.next().?, allocator);
        const test_y = try load.readClassificationDataset(TEST_SAMPLE_SIZE, args.next().?, allocator);
        // const train_x = try read_data_set(train_x_fname, allocator);
        const model = Model{ .X = train_x, .Y = train_y };
        defer model.deinit(allocator);
        const iter_arg = args.next();
        const iterations: u32 = if (iter_arg) |num_str| try std.fmt.parseInt(u32, num_str, 10) else 5000;
        const result = try model.optimize(allocator, iterations, 0.005, true);
        defer result.deinit(allocator);

        const test_model = TestModel{ .X = test_x, .Y = test_y };
        const test_result = try test_model.testParameters(allocator, result.w, result.b);

        try stdout.print("Results: {any}\nAccuracy: {d}\n", .{ test_result.results, test_result.accuracy });
    } else {
        // Less than four arguments provided
        try stdout.print("Error: Expected four arguments\n", .{});
    }
    try bw.flush(); // don't forget to flush!
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
