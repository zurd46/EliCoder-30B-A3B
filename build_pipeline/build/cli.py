import typer
from rich.console import Console
from rich.table import Table

from . import config as cfgmod
from .download import download_base
from .convert_mlx import convert_mlx
from .convert_gguf import convert_gguf, cleanup_f16
from .lm_studio import package_gguf, package_mlx
from .upload import upload_variant, upload_all, whoami
from . import auto as auto_mod
from .prebuilt import fetch_prebuilt

app = typer.Typer(add_completion=False, help="CoderLLM build pipeline — download, quantize, package, upload.")
console = Console()


@app.command("info")
def info():
    c = cfgmod.load()
    t = Table(title=f"{c.display_name}  ({c.model_name})")
    t.add_column("Kind"); t.add_column("Id"); t.add_column("~GB"); t.add_column("Note")
    for q in c.gguf:
        t.add_row("GGUF", q.id, f"{q.expected_size_gb:.1f}", q.description)
    for q in c.mlx:
        t.add_row("MLX", q.id, f"{q.expected_size_gb:.1f}", q.description)
    console.print(t)
    console.print(f"[bold]Base:[/] {c.base_repo}")
    console.print(f"[bold]HF target:[/] {c.hf_owner}/{c.hf_repo_prefix}-<variant>")


@app.command("download")
def cmd_download(force: bool = False):
    download_base(force=force)


@app.command("convert-mlx")
def cmd_convert_mlx(quant: str = typer.Option(None, "--quant"), force: bool = False):
    convert_mlx(quant_id=quant, force=force)


@app.command("convert-gguf")
def cmd_convert_gguf(quant: str = typer.Option(None, "--quant"), force: bool = False):
    convert_gguf(quant_id=quant, force=force)


@app.command("package")
def cmd_package(
    kind: str = typer.Option("all", help="gguf | mlx | all"),
    quant: str = typer.Option(None, "--quant"),
    force: bool = False,
):
    if kind in ("gguf", "all"):
        package_gguf(variant_id=quant, force=force)
    if kind in ("mlx", "all"):
        package_mlx(variant_id=quant, force=force)


@app.command("upload")
def cmd_upload(
    variant: str = typer.Option(None, "--variant", help="single variant id"),
    kind: str = typer.Option(None, help="gguf | mlx — upload all of this kind"),
    private: bool = typer.Option(None),
    dry_run: bool = False,
):
    if variant:
        upload_variant(variant, private=private, dry_run=dry_run)
    else:
        upload_all(kind=kind, private=private, dry_run=dry_run)


@app.command("whoami")
def cmd_whoami():
    whoami()


@app.command("cleanup")
def cmd_cleanup():
    cleanup_f16()


@app.command("detect")
def cmd_detect(target_ctx: int = typer.Option(32768, "--ctx", help="planned context length")):
    h = auto_mod.detect()
    auto_mod.print_plan(h, target_ctx)


@app.command("fetch-prebuilt")
def cmd_fetch_prebuilt(
    kind: str = typer.Option(..., "--kind", help="gguf | mlx"),
    variant: str = typer.Option(..., "--variant"),
    force: bool = False,
):
    fetch_prebuilt(kind, variant, force=force)


@app.command("auto")
def cmd_auto(
    target_ctx: int = typer.Option(32768, "--ctx", help="planned context length"),
    upload: bool = typer.Option(False, help="also upload the picked variant(s) to HF"),
    private: bool = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip confirmation"),
    build_from_source: bool = typer.Option(
        False, "--build-from-source",
        help="force conversion from base BF16 instead of downloading prebuilt quants",
    ),
):
    host = auto_mod.detect()
    picks = auto_mod.print_plan(host, target_ctx)
    if not picks:
        console.print("[red]No suitable variant for this host.[/]")
        raise typer.Exit(1)
    if not yes:
        ok = console.input("\nproceed with these variants? [Y/n] ").strip().lower()
        if ok and ok not in ("y", "yes", "j", "ja"):
            raise typer.Exit(0)

    if not build_from_source:
        console.print("[bold]Strategy:[/] prebuilt-first (fast, community-quality)")
        for kind, vid in picks:
            try:
                fetch_prebuilt(kind, vid)
            except Exception as e:
                console.print(f"[yellow]Prebuilt unavailable for {kind}:{vid} ({e}) — falling back to source build[/]")
                download_base()
                if kind == "mlx":
                    convert_mlx(quant_id=vid)
                else:
                    convert_gguf(quant_id=vid)
            if kind == "mlx":
                package_mlx(variant_id=vid)
            else:
                package_gguf(variant_id=vid)
    else:
        console.print("[bold]Strategy:[/] build-from-source (slow, for fine-tuned bases)")
        download_base()
        for kind, vid in picks:
            if kind == "mlx":
                convert_mlx(quant_id=vid)
                package_mlx(variant_id=vid)
            else:
                convert_gguf(quant_id=vid)
                package_gguf(variant_id=vid)

    if upload:
        for kind, vid in picks:
            upload_variant(vid, private=private)


@app.command("all")
def cmd_all(
    skip_gguf: bool = False,
    skip_mlx: bool = False,
    upload: bool = typer.Option(False, help="also upload to HF"),
    private: bool = typer.Option(None),
):
    download_base()
    if not skip_mlx:
        convert_mlx()
        package_mlx()
    if not skip_gguf:
        convert_gguf()
        package_gguf()
    if upload:
        kind = None
        if skip_gguf: kind = "mlx"
        if skip_mlx:  kind = "gguf"
        upload_all(kind=kind, private=private)


if __name__ == "__main__":
    app()
