"""Bot Telegram de MaintiDoc (livrable L4.1), python-telegram-bot.

Prérequis : un jeton de bot créé avec @BotFather (TELEGRAM_BOT_TOKEN) et le backend
FastAPI démarré (MAINTIDOC_API_URL, BOT_SHARED_SECRET identique des deux côtés).

Lancement :  python -m maintidoc.bot.telegram_bot
"""

from __future__ import annotations

import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from ..config import settings
from . import commands

log = logging.getLogger("maintidoc.bot")
MAX_MESSAGE = 4000  # limite Telegram : 4096 caractères


async def _reply(update: Update, text: str) -> None:
    for start in range(0, len(text), MAX_MESSAGE):
        await update.effective_message.reply_text(text[start:start + MAX_MESSAGE])


def _command(handler):
    async def run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        api: commands.Backend = context.application.bot_data["api"]
        await update.effective_chat.send_action(ChatAction.TYPING)
        try:
            text = await handler(api, update.effective_user.id, context.args or [])
        except Exception:
            log.exception("commande en échec")
            text = "⚠️ Service momentanément indisponible, réessayez plus tard."
        await _reply(update, text)
    return run


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    api: commands.Backend = context.application.bot_data["api"]
    message = update.effective_message
    await message.reply_text("📷 Photo reçue, analyse en cours (plusieurs minutes sur CPU)…")
    await update.effective_chat.send_action(ChatAction.TYPING)
    try:
        file = await message.photo[-1].get_file()
        data = bytes(await file.download_as_bytearray())
        text = await commands.analyze(api, update.effective_user.id, data, (message.caption or "").strip())
    except Exception:
        log.exception("analyse en échec")
        text = "⚠️ Analyse impossible pour le moment."
    await _reply(update, text)


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update, "Utilisez une commande, par exemple /ask <question>, ou /start pour l'aide.")


def build_application(token: str | None = None) -> Application:
    # Traitement concurrent : une analyse de photo longue ne bloque pas les autres utilisateurs.
    application = (Application.builder().token(token or settings.telegram_bot_token)
                   .concurrent_updates(True).build())
    application.bot_data["api"] = commands.Backend()
    for name, handler in {"start": commands.start, "help": commands.start, "ask": commands.ask,
                          "history": commands.history, "tickets": commands.tickets,
                          "create_ticket": commands.create_ticket, "approve": commands.approve,
                          "cancel": commands.cancel}.items():
        application.add_handler(CommandHandler(name, _command(handler)))
    application.add_handler(CommandHandler("analyze", lambda u, c: _reply(
        u, "Envoyez la photo de l'équipement (avec la question en légende).")))
    application.add_handler(MessageHandler(filters.PHOTO, on_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # l'URL contient le jeton du bot
    if not settings.telegram_bot_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN manquant (créez un bot avec @BotFather).")
    if not settings.bot_shared_secret:
        raise SystemExit("BOT_SHARED_SECRET manquant (même valeur que le backend).")
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
