"""Optional user-owned experiment logging; TensorBoard is not a product dependency.

Install TensorBoard in the Python environment you explicitly select. Replace
the synthetic metric with your own training step; see the adaptation guide.
"""

import time


def run(context, *, log_dir):
    # Import only when the user actually runs this example.
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    cursor = int(context.resume_state.get("cursor", 0)) if context.resume_state else 0
    writer = EventFileWriter(log_dir)
    try:
        while context.remaining_steps and not context.stop_requested:
            cursor += 1
            metric = 1.0 / cursor  # Replace with a real, finite numeric metric.
            writer.add_event(Event(wall_time=time.time(), step=cursor,
                summary=Summary(value=[Summary.Value(tag="example/loss", simple_value=metric)])))
            writer.flush()
            # JSON state tracks this example's cursor only; it does not save a
            # model, optimizer, data loader, or TensorBoard's own state.
            if not context.report_step({"loss": metric}, state={"cursor": cursor}):
                break
    finally:
        writer.close()
    return {"cursor": cursor}
