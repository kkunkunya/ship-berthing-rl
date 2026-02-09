def compute_failure_flags(
    position_error,
    heading_error,
    velocity,
    yaw_rate,
    position_tolerance,
    heading_tolerance,
    velocity_tolerance,
    yaw_rate_tolerance,
):
    return {
        "position": position_error > position_tolerance,
        "heading": heading_error > heading_tolerance,
        "velocity": velocity > velocity_tolerance,
        "yaw_rate": yaw_rate > yaw_rate_tolerance,
    }


def compute_success(
    position_error,
    heading_error,
    velocity,
    yaw_rate,
    position_tolerance,
    heading_tolerance,
    velocity_tolerance,
    yaw_rate_tolerance,
):
    flags = compute_failure_flags(
        position_error=position_error,
        heading_error=heading_error,
        velocity=velocity,
        yaw_rate=yaw_rate,
        position_tolerance=position_tolerance,
        heading_tolerance=heading_tolerance,
        velocity_tolerance=velocity_tolerance,
        yaw_rate_tolerance=yaw_rate_tolerance,
    )
    return not any(flags.values())
