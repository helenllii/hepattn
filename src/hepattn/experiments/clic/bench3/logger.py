"""A CSVLogger that tolerates hepattn's CLI logger wiring.

`hepattn.utils.cli.CLI.before_instantiate_classes` sets `trainer.logger.init_args.offline_directory` on every `fit`,
assuming the Comet logger used in production. Lightning's CSVLogger has no such argument, so a bench3 run with a plain
CSVLogger fails at instantiation (job 7835338). This subclass accepts and ignores it; logging is otherwise unchanged.
The class keeps the name `CSVLogger` so the CLI does not add a second, redundant CSV logger after instantiation.
"""

from lightning.pytorch.loggers import CSVLogger as _LightningCSVLogger


class CSVLogger(_LightningCSVLogger):
    def __init__(
        self,
        save_dir: str = "logs",
        name: str | None = "lightning_logs",
        version: int | str | None = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
        offline_directory: str | None = None,  # noqa: ARG002 - injected by the hepattn CLI, unused by CSV logging
    ) -> None:
        super().__init__(save_dir=save_dir, name=name, version=version, prefix=prefix, flush_logs_every_n_steps=flush_logs_every_n_steps)
