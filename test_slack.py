import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()

token = os.getenv("SLACK_USER_TOKEN")

if not token:
    raise RuntimeError("Falta SLACK_USER_TOKEN en .env")

client = WebClient(token=token)


def conv_type(c):
    if c.get("is_im"):
        return "DM"
    if c.get("is_mpim"):
        return "MPIM"
    if c.get("is_private"):
        return "Privado"
    return "Público"


def conv_name(c):
    return c.get("name") or c.get("user") or c.get("id")


def list_with_conversations_list(types):
    print(f"\n=== conversations.list: {types} ===")
    cursor = None
    total = 0
    page = 0

    while True:
        page += 1
        resp = client.conversations_list(
            types=types,
            limit=200,
            cursor=cursor,
            exclude_archived=True,
        )

        channels = resp.get("channels", [])
        total += len(channels)

        print(f"Página {page}: {len(channels)} resultados")

        for c in channels[:15]:
            print(f"- [{conv_type(c)}] {conv_name(c)} | {c.get('id')}")

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    print(f"TOTAL {types}: {total}")
    return total


def list_with_users_conversations(types, user_id):
    print(f"\n=== users.conversations: {types} ===")
    cursor = None
    total = 0
    page = 0

    while True:
        page += 1
        resp = client.users_conversations(
            user=user_id,
            types=types,
            limit=200,
            cursor=cursor,
            exclude_archived=True,
        )

        channels = resp.get("channels", [])
        total += len(channels)

        print(f"Página {page}: {len(channels)} resultados")

        for c in channels[:15]:
            print(f"- [{conv_type(c)}] {conv_name(c)} | {c.get('id')}")

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    print(f"TOTAL {types}: {total}")
    return total


try:
    auth = client.auth_test()
    user_id = auth.get("user_id")

    print("✅ Token válido")
    print("Workspace:", auth.get("team"))
    print("Usuario:", auth.get("user"))
    print("User ID:", user_id)

    print("\n--- PRUEBA POR TIPO CON conversations.list ---")
    for t in ["public_channel", "private_channel", "im", "mpim"]:
        try:
            list_with_conversations_list(t)
        except SlackApiError as e:
            print(f"❌ Error en conversations.list {t}: {e.response.get('error')}")
            print(e.response.data)

    print("\n--- PRUEBA POR TIPO CON users.conversations ---")
    for t in ["public_channel", "private_channel", "im", "mpim"]:
        try:
            list_with_users_conversations(t, user_id)
        except SlackApiError as e:
            print(f"❌ Error en users.conversations {t}: {e.response.get('error')}")
            print(e.response.data)

except SlackApiError as e:
    print("❌ Error de Slack")
    print("Código:", e.response.get("error"))
    print("Detalle:", e.response.data)

except Exception as e:
    print("❌ Error general")
    print(str(e))
