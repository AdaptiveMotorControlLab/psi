from psi.data.tasks.mcqa import MCQATask


class ARCTask(MCQATask):
    """The ARC dataset is a multiple-choice question answering dataset for grade-school science exams."""

    task_name = "arc"

    @property
    def position_names(self) -> tuple[str, ...]:
        """Return the names of the cut positions in this task."""
        # NOTE: Filter out the `correct_symb_period_positions` from
        # the MCQA task, as it is not relevant for the ARC task.
        return ('correct_symbol', "last_token")