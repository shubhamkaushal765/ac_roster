import numpy as np
import plotly.graph_objects as go


class Plotter:
    """
    A class to handle all plotting and visualization functionality for officer scheduling.
    """

    def __init__(self, num_slots=48, num_counters=41, start_hour=10):
        """
        Initialize the Plotter with configuration parameters.

        Args:
            num_slots (int): Number of time slots (default: 48)
            num_counters (int): Number of counters (default: 41)
            start_hour (int): Starting hour for the schedule (default: 10)
        """
        self.num_slots = num_slots
        self.num_counters = num_counters
        self.start_hour = start_hour

    def slot_to_hhmm(self, slot: int) -> str:
        """
        Convert slot index back to hhmm string.

        Args:
            slot (int): Slot index (0-47)

        Returns:
            str: Time in HHMM format
        """
        h = self.start_hour + slot // 4
        m = (slot % 4) * 15
        return f"{h:02d}{m:02d}"

    def plot_officer_timetable_with_labels(self, counter_matrix):
        """
        Plots an interactive timetable with officer IDs inside each cell.
        Consecutive cells with the same officer are merged with one label and thick border.

        Parameters:
            counter_matrix: 2D numpy array or list of lists
                Shape: (NUM_COUNTERS, NUM_SLOTS)
                counter_matrix[i, j] = officer at counter i, time_slot j

        Returns:
            fig: Plotly Figure object
        """
        counter_matrix = np.array(counter_matrix, dtype=object)
        counter_matrix[counter_matrix == 0] = "0"
        num_counters, time_slots = counter_matrix.shape

        # Create numeric matrix for colors
        color_matrix = np.zeros((num_counters, time_slots), dtype=int)
        for i in range(num_counters):
            for j in range(time_slots):
                val = str(counter_matrix[i, j])
                if val.startswith("M"):
                    color_matrix[i, j] = 1
                elif val.startswith("S"):
                    color_matrix[i, j] = 2
                elif val.startswith("OT"):
                    color_matrix[i, j] = 3
                else:
                    color_matrix[i, j] = 0

        # Generate x-axis labels using slot_to_hhmm
        x_labels = [self.slot_to_hhmm(t) for t in range(time_slots)]

        # Create heatmap with proper hover text
        hover_text = [
            [
                f"Time: {x_labels[j]}<br>Counter: C{i + 1}<br>Officer: {counter_matrix[i, j]}"
                for j in range(time_slots)
            ]
            for i in range(num_counters)
        ]

        heatmap = go.Heatmap(
            z=color_matrix,
            y=[f"C{t + 1}" for t in range(num_counters)],
            x=list(range(time_slots)),
            text=hover_text,
            hoverinfo="text",
            showscale=False,
            colorscale=[
                [0, "#2E2C2C"],  # Unassigned
                [0.33, "#a2d2ff"],  # M-type
                [0.66, "#ffc6d9"],  # S-type
                [1, "#FDFD96"],  # OT-type
            ],
            zmin=0,
            zmax=3,
            opacity=0.85
        )

        # Find merged regions (consecutive horizontal cells with same officer)
        annotations = []
        shapes = []

        for i in range(num_counters):
            j = 0
            while j < time_slots:
                officer = str(counter_matrix[i, j])
                if officer != "0":
                    # Find the end of this consecutive block
                    j_end = j
                    while j_end < time_slots and str(
                            counter_matrix[i, j_end]
                    ) == officer:
                        j_end += 1

                    # Add annotation at the center of the merged region
                    center_x = (j + j_end - 1) / 2
                    annotations.append(
                        dict(
                            x=center_x,
                            y=i,
                            text=officer,
                            showarrow=False,
                            font=dict(color="black", size=18),
                        )
                    )

                    # Add border around the merged region
                    shapes.append(
                        dict(
                            type="rect",
                            x0=j - 0.5,
                            x1=j_end - 0.5,
                            y0=i - 0.5,
                            y1=i + 0.5,
                            line=dict(color="black", width=1),
                            fillcolor="rgba(0,0,0,0)",
                        )
                    )

                    j = j_end
                else:
                    j += 1

        # Add graph-paper style grid lines ONLY in unassigned cells
        grid_shapes = []
        for x in range(time_slots + 1):
            # Check each counter segment to see if we should draw the line
            for i in range(num_counters):
                # Determine if this segment should have a grid line
                # Only draw if both adjacent cells (or edge) are unassigned
                left_unassigned = (x == 0) or (
                        x > 0 and color_matrix[i, x - 1] == 0)
                right_unassigned = (x == time_slots) or (
                        x < time_slots and color_matrix[i, x] == 0)

                if left_unassigned and right_unassigned:
                    # Every 4 slots → thick solid line
                    if x % 4 == 0:
                        grid_shapes.append(
                            dict(
                                type="line",
                                x0=x - 0.5,
                                x1=x - 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                line=dict(
                                    color="rgba(150,150,150,0.6)", width=1
                                ),
                                layer='above'
                            )
                        )
                    # Every 2 slots → dashed line
                    elif x % 2 == 0:
                        grid_shapes.append(
                            dict(
                                type="line",
                                x0=x - 0.5,
                                x1=x - 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                line=dict(
                                    color="rgba(120,120,120,0.4)", width=2,
                                    dash="dash"
                                ),
                                layer='above'
                            )
                        )

        # Combine all shapes
        all_shapes = shapes + grid_shapes

        # Create figure
        fig = go.Figure(data=[heatmap])
        fig.update_layout(
            title="Officer Timetable",
            xaxis_title="Time",
            yaxis_title="Counter",
            annotations=annotations,
            shapes=all_shapes,
            yaxis_autorange="reversed",
            dragmode=False,
            autosize=True,
            height=900,
            width=900
        )

        # Show ticks with time labels
        fig.update_xaxes(
            tickvals=list(range(0, time_slots, 4)),
            ticktext=[x_labels[i] for i in range(0, time_slots, 4)],
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            type='linear',
            ticks=''
        )
        fig.update_yaxes(showgrid=False, showticklabels=True, zeroline=False)

        return fig

    def plot_officer_schedule_with_labels(self, officer_schedule):
        """
        Plots an interactive timetable showing each officer's assigned counter at each time slot.
        Consecutive same-counter assignments are merged and bordered.

        Parameters:
            officer_schedule (dict):
                Keys = officer IDs (e.g., 'M1', 'S2', ...)
                Values = list of counter numbers (int) for each time slot (0 = unassigned)

        Returns:
            fig: plotly.graph_objects.Figure
        """
        officers = list(officer_schedule.keys())
        num_officers = len(officers)
        num_slots = len(next(iter(officer_schedule.values())))

        # Build numeric color matrix for background color coding
        color_matrix = np.zeros((num_officers, num_slots), dtype=int)
        label_matrix = np.empty((num_officers, num_slots), dtype=object)

        for i, officer_id in enumerate(officers):
            for t, counter in enumerate(officer_schedule[officer_id]):
                officer_str = str(officer_id)
                if counter != 0:
                    color_matrix[i, t] = (
                        1
                        if officer_str.startswith("M")
                        else 2
                        if officer_str.startswith("S")
                        else 3
                    )
                    label_matrix[i, t] = f"C{counter}"
                else:
                    color_matrix[i, t] = 0
                    label_matrix[i, t] = ""

        # Generate x-axis labels using slot_to_hhmm
        x_labels = [self.slot_to_hhmm(t) for t in range(num_slots)]

        # Base heatmap with numeric x-axis
        heatmap = go.Heatmap(
            z=color_matrix,
            y=officers,
            x=list(range(num_slots)),  # Use numeric values
            showscale=False,
            colorscale=[
                [0, "#2E2C2C"],  # Unassigned
                [0.33, "#a2d2ff"],  # M-type
                [0.66, "#ffc6d9"],  # S-type
                [1, "#FDFD96"],  # OT-type
            ],
            zmin=0,
            zmax=3,
            opacity=0.85
        )

        annotations = []
        shapes = []

        # Merge consecutive same-counter cells horizontally
        for i, officer_id in enumerate(officers):
            t = 0
            while t < num_slots:
                counter = officer_schedule[officer_id][t]
                if counter != 0:
                    t_end = t
                    while (
                            t_end < num_slots and officer_schedule[officer_id][
                        t_end] == counter
                    ):
                        t_end += 1

                    # Add merged label at center of block
                    center_x = (t + t_end - 1) / 2
                    annotations.append(
                        dict(
                            x=center_x,
                            y=officer_id,
                            text=f"C{counter}",
                            showarrow=False,
                            font=dict(color="black", size=18),
                        )
                    )

                    # Add thick border around merged region
                    shapes.append(
                        dict(
                            type="rect",
                            x0=t - 0.5,
                            x1=t_end - 0.5,
                            y0=i - 0.5,
                            y1=i + 0.5,
                            line=dict(color="black", width=1),
                            fillcolor="rgba(0,0,0,0)",
                        )
                    )

                    t = t_end
                else:
                    t += 1

        # Add graph-paper style grid lines ONLY in unassigned cells
        grid_shapes = []
        for x in range(num_slots + 1):
            # Check each officer segment to see if we should draw the line
            for i in range(num_officers):
                # Determine if this segment should have a grid line
                # Only draw if both adjacent cells (or edge) are unassigned
                left_unassigned = (x == 0) or (
                        x > 0 and color_matrix[i, x - 1] == 0)
                right_unassigned = (x == num_slots) or (
                        x < num_slots and color_matrix[i, x] == 0)

                if left_unassigned and right_unassigned:
                    # Every 4 slots → thick solid line
                    if x % 4 == 0:
                        grid_shapes.append(
                            dict(
                                type="line",
                                x0=x - 0.5,
                                x1=x - 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                line=dict(
                                    color="rgba(150,150,150,0.6)", width=1
                                ),
                                layer='above'
                            )
                        )
                    # Every 2 slots → dashed line
                    elif x % 2 == 0:
                        grid_shapes.append(
                            dict(
                                type="line",
                                x0=x - 0.5,
                                x1=x - 0.5,
                                y0=i - 0.5,
                                y1=i + 0.5,
                                line=dict(
                                    color="rgba(120,120,120,0.4)", width=2,
                                    dash="dash"
                                ),
                                layer='above'
                            )
                        )

        # Combine all shapes
        all_shapes = shapes + grid_shapes

        # Build figure
        fig = go.Figure(data=[heatmap])
        fig.update_layout(
            title="Officer Timetable (Counter Assignments)",
            xaxis_title="Time",
            yaxis_title="Officer",
            annotations=annotations,
            shapes=all_shapes,
            yaxis_autorange="reversed",
            dragmode=False,
            autosize=True,
            height=900,
            width=900
        )

        # Show ticks with time labels
        fig.update_xaxes(
            tickvals=list(range(0, num_slots, 4)),
            ticktext=[x_labels[i] for i in range(0, num_slots, 4)],
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            type='linear',
            ticks=''
        )
        fig.update_yaxes(showgrid=False, showticklabels=True, zeroline=False)

        return fig
