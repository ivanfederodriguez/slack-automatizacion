import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from rich import print
from rich.markup import escape

from replay import format_reprocess_message_result, run_reprocess_message_fixture
from slack_personal_agent import (
    AgentApp,
    AgentConfig,
    ConfigError,
    NoSlackSideEffectsAgentApp,
    format_reprocess_open_trello_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slack Personal Agent")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "bootstrap",
            "once",
            "poll",
            "brief",
            "review",
            "approve-reply",
            "tasks",
            "trello-auth-url",
            "trello-lists",
            "trello-sync",
            "trello-reply-sync",
            "trello-waiting-sync",
            "trello-done-sync",
            "telegram-poll",
            "transcribe-audio",
            "transcribe-audio-folder",
            "reprocess-message",
            "reprocess-open-trello",
            "install-autostart",
            "uninstall-autostart",
        ],
        help="Qué acción ejecutar.",
    )
    parser.add_argument(
        "task_id",
        nargs="?",
        help="ID de tarea para comandos puntuales o PATH para comandos de transcripción.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad máxima de tareas a mostrar con `tasks` o por sección en `brief`.",
    )
    parser.add_argument(
        "--reply",
        default=None,
        help="Texto manual para aprobar con `approve-reply`; si se omite usa el borrador guardado.",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Con `approve-reply`, además de aprobar envía la respuesta en Slack.",
    )
    parser.add_argument(
        "--send-approved-replies",
        action="store_true",
        help="Con `review`, aprobar o editar también envía la respuesta en Slack.",
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="Fixture JSON para `reprocess-message`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Con `reprocess-message` o `reprocess-open-trello`, no modifica DB/Trello ni envía mensajes.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Con `reprocess-open-trello`, aplica cambios en SQLite y Trello.",
    )
    parser.add_argument(
        "--show-before-after",
        action="store_true",
        help="Con `reprocess-message`, muestra clasificación del modelo y clasificación final.",
    )
    parser.add_argument(
        "--only-salesforce",
        action="store_true",
        help="Con `reprocess-open-trello`, procesa solo tareas que quedan relacionadas con Salesforce.",
    )
    parser.add_argument(
        "--only-trello-created",
        action="store_true",
        help="Con `reprocess-open-trello`, procesa solo tareas con card Trello existente.",
    )
    parser.add_argument(
        "--include-waiting",
        action="store_true",
        help="Con `reprocess-open-trello`, incluye tareas en espera de información.",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command in {"reprocess-message", "reprocess-open-trello"}:
            env = dict(os.environ)
            env.setdefault("SLACK_USER_TOKEN", "xoxp-replay-local-token")
            config = AgentConfig.from_env(env)
        else:
            config = AgentConfig.from_env()
        app = NoSlackSideEffectsAgentApp(config) if args.command == "reprocess-open-trello" else AgentApp(config)
    except ConfigError as exc:
        print(f"[red]{exc}[/red]")
        return 1

    if args.command == "doctor":
        return 0 if app.doctor() else 1
    if args.command == "bootstrap":
        app.bootstrap()
        return 0
    if args.command == "once":
        app.poll_once()
        return 0
    if args.command == "poll":
        app.loop()
        return 0
    if args.command == "brief":
        app.print_brief(limit=args.limit if args.limit is not None else 5)
        return 0
    if args.command == "review":
        app.print_review(
            limit=args.limit if args.limit is not None else 10,
            send_replies=args.send_approved_replies,
        )
        return 0
    if args.command == "approve-reply":
        if args.task_id is None:
            parser.error("approve-reply requiere el ID de la tarea.")
        task_id = int(args.task_id)
        ok = app.approve_reply(task_id, args.reply, send=args.send)
        if ok:
            action = "aprobada y enviada" if args.send else "aprobada"
            print(f"[green]Respuesta {action} para tarea #{task_id}.[/green]")
            return 0
        return 1
    if args.command == "tasks":
        app.print_tasks(limit=args.limit if args.limit is not None else 20)
        return 0
    if args.command == "trello-auth-url":
        print(app.trello_auth_url())
        return 0
    if args.command == "trello-lists":
        app.print_trello_lists()
        return 0
    if args.command == "trello-sync":
        synced = app.sync_pending_trello_tasks(limit=args.limit if args.limit is not None else 20)
        print(f"[green]Cards creadas en Trello:[/green] {synced}")
        return 0
    if args.command == "trello-reply-sync":
        sent = app.sync_trello_reply_commands(limit=args.limit if args.limit is not None else 50)
        print(f"[green]Respuestas enviadas a Slack desde Trello:[/green] {sent}")
        return 0
    if args.command == "trello-waiting-sync":
        requested = app.sync_trello_waiting_requests(limit=args.limit if args.limit is not None else 50)
        print(f"[green]Pedidos de información enviados a Slack:[/green] {requested}")
        return 0
    if args.command == "trello-done-sync":
        synced = app.sync_trello_done_tasks(limit=args.limit if args.limit is not None else 50)
        print(f"[green]Tareas cerradas o marcadas como done_pending_reply:[/green] {synced}")
        return 0
    if args.command == "telegram-poll":
        handled = app.poll_telegram_updates(limit=args.limit if args.limit is not None else 20)
        print(f"[green]Comandos Telegram procesados:[/green] {handled}")
        return 0
    if args.command == "transcribe-audio":
        if args.task_id is None:
            parser.error("transcribe-audio requiere un PATH.")
        print(app.transcribe_audio_path(Path(args.task_id)))
        return 0
    if args.command == "transcribe-audio-folder":
        if args.task_id is None:
            parser.error("transcribe-audio-folder requiere un PATH.")
        for path, transcript in app.transcribe_audio_folder(Path(args.task_id)):
            print(f"\n[bold]{path.name}[/bold]")
            print(transcript)
        return 0
    if args.command == "reprocess-message":
        if not args.fixture:
            parser.error("reprocess-message requiere --fixture.")
        result = run_reprocess_message_fixture(
            config=config,
            fixture_path=Path(args.fixture),
            dry_run=args.dry_run,
        )
        print(escape(format_reprocess_message_result(result, show_before_after=args.show_before_after)))
        return 0
    if args.command == "reprocess-open-trello":
        if args.dry_run and args.apply:
            parser.error("reprocess-open-trello no acepta --dry-run y --apply juntos.")
        result = app.reprocess_open_trello_tasks(
            apply=args.apply,
            limit=args.limit,
            only_salesforce=args.only_salesforce,
            only_trello_created=args.only_trello_created,
            include_waiting=args.include_waiting,
        )
        print(escape(format_reprocess_open_trello_result(result)))
        return 0 if result.errors == 0 else 1
    if args.command == "install-autostart":
        app.install_autostart()
        return 0
    if args.command == "uninstall-autostart":
        app.uninstall_autostart()
        return 0

    parser.error("Comando no soportado.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
