# Slack Personal Agent

Agente personal para Slack que:

- monitorea DMs, group DMs y canales privados;
- decide si un mensaje requiere accion;
- genera una clasificacion estructurada y un borrador de respuesta;
- guarda tareas en SQLite;
- opcionalmente crea cards en Trello;
- permite revisar, aprobar y enviar respuestas solo con confirmacion explicita.

La regla operativa importante es esta:

> nada se envia a Slack sin una accion explicita de Ivan.

## Estado actual

El CLI ya cubre una v0 util para uso personal:

- `doctor` valida Slack, modelo y acceso general;
- `bootstrap` registra conversaciones y fija un baseline sin procesar historial viejo;
- `once` hace un ciclo de polling;
- `poll` deja el agente corriendo;
- `brief` resume tareas abiertas y respuestas aprobadas sin enviar;
- `review` permite aprobar, editar, ignorar, snoozear o marcar done;
- `approve-reply` aprueba una respuesta puntual y opcionalmente la envia;
- `trello-*` cubre auth, listado de listas y resincronizacion;
- `install-autostart` y `uninstall-autostart` manejan launch agents locales.

## Requisitos

- Python 3
- un token de Slack valido en `SLACK_USER_TOKEN`
- Ollama local o Groq, segun `MODEL_PROVIDER`
- Trello opcional

El token de Slack que uses debe poder llamar, como minimo, a estos metodos que el codigo usa hoy:

- `auth.test`
- `users.info`
- `conversations.list`
- `conversations.history`
- `conversations.replies`
- `chat.postMessage`

Si `chat.postMessage` falla con algo como `missing_scope` o `not_allowed_token_type`, el agente deja auditado el error en `reply_error` y no marca la tarea como respondida.

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

DB_PATH=slack_agent.db
```

Notas utiles:

- el agente monitorea `private_channel`, `im` y `mpim`;
- los canales publicos quedan fuera de alcance por ahora;
- la base local por defecto es `slack_agent.db`;
- si `TRELLO_ENABLED=false`, todo el flujo principal sigue funcionando sin Trello.

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

Estados utiles:

- `new`
- `reply_approved`
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
- review interactivo;
- aprobacion de respuestas;
- envio exitoso y fallo de envio a Slack.

## Lo que no hace todavia

- no responde automaticamente por defecto;
- no procesa canales publicos;
- no resuelve permisos de Slack por si solo;
- no intenta enviar si no hay texto aprobado o borrador disponible.
