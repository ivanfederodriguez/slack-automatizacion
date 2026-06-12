# Slack Personal Agent

Agente personal para Slack que:

- monitorea DMs, group DMs y canales privados;
- decide si un mensaje requiere accion;
- genera una clasificacion estructurada y un borrador de respuesta;
- guarda tareas en SQLite;
- opcionalmente crea cards en Trello;
- acusa recibo y suma contexto automaticamente en el thread del pedido;
- permite revisar, aprobar y enviar respuestas finales solo con confirmacion explicita;
- usa Telegram como canal privado de aprobacion cuando Trello marca una tarea como hecha.

La regla operativa importante es esta:

> solo se envian automaticamente acuses de recibo y confirmaciones de contexto. Las respuestas finales o explicaciones de solucion requieren una accion explicita de Ivan.

## Estado actual

El CLI ya cubre una v0 util para uso personal:

- `doctor` valida Slack, modelo y acceso general;
- `bootstrap` registra conversaciones y fija un baseline sin procesar historial viejo;
- `once` hace un ciclo de polling;
- `poll` deja el agente corriendo;
- `brief` resume tareas abiertas y respuestas aprobadas sin enviar;
- `review` permite aprobar, editar, ignorar, snoozear o marcar done;
- `approve-reply` aprueba una respuesta puntual y opcionalmente la envia;
- `trello-*` cubre auth, listado de listas, resincronizacion y deteccion de cards hechas;
- `telegram-poll` procesa comandos de aprobacion enviados por Ivan;
- `install-autostart` y `uninstall-autostart` manejan launch agents locales.

## Requisitos

- Python 3
- un token de Slack valido en `SLACK_USER_TOKEN`
- Ollama local o Groq, segun `MODEL_PROVIDER`
- Trello opcional
- Telegram opcional para aprobacion privada

El token de Slack que uses debe poder llamar, como minimo, a estos metodos que el codigo usa hoy:

- `auth.test`
- `users.info`
- `conversations.list`
- `conversations.history`
- `conversations.replies`
- `chat.postMessage`

Si `chat.postMessage` falla con algo como `missing_scope` o `not_allowed_token_type`, el agente deja auditado el error en `ack_error`, `context_ack_error` o `reply_error` segun el paso afectado, y no marca la tarea como respondida cuando se trata de una respuesta final.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Completa `.env` con tus credenciales. Para un arranque seguro, deja esto asi:

```bash
SLACK_SEND_APPROVED_REPLIES=false
```

## Configuracion

Variables principales:

```bash
SLACK_USER_TOKEN=
MY_SLACK_USER_ID=
MY_MENTION_ALIASES=ivan,ivo

MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_BASE_URL=http://127.0.0.1:11434

# fallback opcional
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-20b

POLL_SECONDS=300
SLACK_SLEEP_SECONDS=1.2
INCLUDE_SELF_FOR_TEST=false
SLACK_SEND_APPROVED_REPLIES=false

# Trello opcional
TRELLO_ENABLED=false
TRELLO_AUTO_CREATE=true
TRELLO_API_KEY=
TRELLO_TOKEN=
TRELLO_LIST_ID=
TRELLO_MEMBER_IDS=
TRELLO_LABEL_IDS=
TRELLO_CARD_POSITION=top
TRELLO_DONE_MODE=check
TRELLO_DONE_CHECKLIST_ITEM_NAME=Hecho
TRELLO_DONE_LIST_ID=
TRELLO_DONE_LIST_NAMES=Hecho,Done

# Telegram opcional
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Audio opcional
AUDIO_TRANSCRIPTION_ENABLED=true
SLACK_AUDIO_TRANSCRIPTS_ENABLED=true
LOCAL_WHISPER_ENABLED=true
LOCAL_WHISPER_MODEL=tiny
LOCAL_WHISPER_LANGUAGE=es
LOCAL_WHISPER_DEVICE=auto
LOCAL_WHISPER_COMPUTE_TYPE=auto
LOCAL_WHISPER_MAX_SECONDS=600
LOCAL_WHISPER_KEEP_AUDIO_FILES=false
LOCAL_WHISPER_CACHE_DIR=~/Library/Application Support/slack-personal-agent/audio
AUDIO_TRANSCRIPT_FUSION_ENABLED=true
AUDIO_TRANSCRIPT_FUSION_MODEL=main

DB_PATH=slack_agent.db
```

Notas utiles:

- el agente monitorea `private_channel`, `im` y `mpim`;
- los canales publicos quedan fuera de alcance por ahora;
- la base local por defecto es `slack_agent.db`;
- si `TRELLO_ENABLED=false`, todo el flujo principal sigue funcionando sin Trello;
- si `TELEGRAM_ENABLED=false`, las tareas hechas quedan en `done_pending_reply` con error auditado hasta configurar Telegram o cerrarlas manualmente.

## Flujo conversacional

Cuando detecta una tarea accionable nueva, el agente responde en el thread original:

```text
Dale, lo tomo. Lo registre como: '{summary}'. Si queres, podes agregar contexto en este mismo hilo.
```

En DM no menciona al solicitante. En canales privados y group DMs antepone `<@user_id>`.

Si llega contexto nuevo en el mismo thread, actualiza la tarea existente, agrega comentario en Trello si la card existe y confirma:

```text
Buenisimo, gracias. Lo sumo al pedido. El registro queda como: '{summary_actualizado}'.
```

Cuando una card queda marcada como hecha segun `TRELLO_DONE_MODE`, la tarea cambia a `done_pending_reply`. El modo recomendado es `check`, que usa el check nativo de Trello (`dueComplete`). Tambien siguen disponibles `list`, `checklist` y `list_or_check` para compatibilidad.

El agente no contesta Slack al detectar una card hecha: prepara un cierre seguro y manda Telegram a Ivan con:

```text
/send TASK_ID
/edit TASK_ID texto
/nosend TASK_ID
```

## Configurar Telegram

Para habilitar la aprobacion privada por Telegram:

1. Crear un bot con `@BotFather` y guardar el token en `TELEGRAM_BOT_TOKEN`.
2. Abrir un chat con ese bot y mandarle al menos un mensaje manual.
3. Obtener el `chat_id` y cargarlo en `TELEGRAM_CHAT_ID`.
4. Activar `TELEGRAM_ENABLED=true`.

Una forma simple de obtener `TELEGRAM_CHAT_ID` es abrir esta URL despues de haberle escrito al bot:

```text
https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates
```

En la respuesta JSON aparece `message.chat.id`. Ese valor es el que va en `TELEGRAM_CHAT_ID`.

Si usas un chat 1:1 con el bot, el `chat_id` suele ser un entero positivo. Si usas un grupo privado, puede venir como entero negativo. En ambos casos funciona mientras coincida con el chat donde Ivan va a aprobar.

## Comandos operativos

`python main.py trello-done-sync --limit 50`

Chequea cards ya creadas en Trello y, si alguna paso a una lista marcada como hecha, mueve la tarea a `done_pending_reply` y dispara la notificacion de Telegram. Sirve para probar el flujo manualmente o para resincronizar si el loop no estuvo corriendo.

`python main.py telegram-poll --limit 20`

Lee comandos pendientes del bot de Telegram y procesa `/send`, `/edit` y `/nosend`. Es util para probar aprobaciones manualmente o para destrabar mensajes si queres correr Telegram por separado.

Las respuestas finales no se envian automaticamente. El agente solo manda el mensaje final a Slack cuando Ivan usa `/send TASK_ID` o cuando se dispara un envio manual explicito por el flujo seguro existente.

## Audio

Antes de clasificar un mensaje, el agente puede convertir audios de Slack en texto. Primero usa la transcripcion que ya venga en Slack si existe; si `LOCAL_WHISPER_ENABLED=true`, descarga el archivo privado con el token de Slack y lo transcribe con Whisper local bajo demanda. Si ambas versiones existen, las fusiona con el modelo principal usando un prompt conservador que no inventa contenido.

Si solo hay una transcripcion disponible, usa esa. Si no hay ninguna o falla la descarga/transcripcion, lo deja auditado en SQLite y no rompe el procesamiento. El agente no descarga audios cuando `LOCAL_WHISPER_ENABLED=false`, y no conserva archivos salvo que `LOCAL_WHISPER_KEEP_AUDIO_FILES=true`.

Scopes utiles para audio en Slack: el token debe poder leer archivos privados, normalmente con `files:read` ademas de los scopes de conversaciones que ya usa el agente.

Para probar Whisper local sin depender de Slack:

```bash
python main.py transcribe-audio /ruta/audio.m4a
python main.py transcribe-audio-folder /ruta/carpeta
```

## Primer arranque

1. Verifica credenciales y modelo:

```bash
python main.py doctor
```

2. Registra conversaciones y arranca desde ahora:

```bash
python main.py bootstrap
```

`bootstrap` no procesa historial viejo. Solo guarda el estado base para leer mensajes nuevos en adelante.

3. Corre un ciclo manual:

```bash
python main.py once
```

4. Si queres dejarlo corriendo:

```bash
python main.py poll
```

## Flujo seguro de respuestas

El flujo recomendado es:

1. Detectar tareas:

```bash
python main.py brief --limit 10
```

2. Revisar sin enviar:

```bash
python main.py review --limit 5
```

Acciones del review:

- `a` aprueba el borrador
- `e` edita y aprueba
- `i` ignora
- `t` abre Trello
- `s` snooze
- `d` marca done
- `enter` salta
- `q` sale

3. Confirmar que quedo aprobada pero sin enviar:

```bash
python main.py brief --limit 10
```

4. Enviar solo una tarea puntual, con confirmacion explicita:

```bash
python main.py approve-reply 123 --send
```

5. Cuando ya confies en el flujo, podes revisar y enviar en el mismo paso:

```bash
python main.py review --limit 10 --send-approved-replies
```

Si `SLACK_SEND_APPROVED_REPLIES=true`, `review` entra en modo envio aunque no pases el flag. Por eso el valor recomendado por defecto es `false`.

## Comandos

```bash
python main.py doctor
python main.py bootstrap
python main.py once
python main.py poll
python main.py brief --limit 10
python main.py review --limit 10
python main.py review --limit 10 --send-approved-replies
python main.py approve-reply 123
python main.py approve-reply 123 --reply "Texto manual"
python main.py approve-reply 123 --send
python main.py tasks --limit 20
python main.py trello-auth-url
python main.py trello-lists
python main.py trello-sync --limit 20
python main.py trello-done-sync --limit 50
python main.py telegram-poll --limit 20
python main.py transcribe-audio /ruta/audio.m4a
python main.py transcribe-audio-folder /ruta/carpeta
python main.py install-autostart
python main.py uninstall-autostart
```

## Auditoria y estados

La base ya guarda informacion para auditar el flujo de respuesta:

- `reply_approved_at`
- `reply_sent_at`
- `reply_ts`
- `reply_error`
- `manual_reply`
- `case_key`
- `requester_user_id`
- `requester_label`
- `thread_ts`
- `acknowledged_at`
- `last_context_ack_at`
- `done_pending_reply_at`
- `final_reply_suggestion`
- `telegram_error`

La tabla `audio_transcriptions` guarda la auditoria de audios: transcript de Slack, transcript local, transcript fusionado, texto seleccionado, estado y error si aplica.

Estados utiles:

- `new`
- `reply_approved`
- `done_pending_reply`
- `responded`
- `ignored`
- `done`
- `snoozed`

`brief` separa las tareas que necesitan respuesta de las respuestas aprobadas sin enviar.

## Tests

```bash
pytest
```

La suite actual cubre, entre otras cosas:

- clasificacion y relevancia;
- enrichment de URLs;
- sync a Trello;
- acuse automatico y actualizacion de contexto;
- deteccion de cards hechas y aprobacion Telegram;
- transcripcion de audio Slack/local y fusion conservadora;
- review interactivo;
- aprobacion de respuestas;
- envio exitoso y fallo de envio a Slack.

## Lo que no hace todavia

- no envia respuestas finales automaticamente;
- no procesa canales publicos;
- no resuelve permisos de Slack por si solo;
- no intenta enviar si no hay texto aprobado o borrador disponible.
