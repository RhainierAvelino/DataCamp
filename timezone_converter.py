"""Let a model call a local Python function to convert between time zones."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import OpenAI


client = OpenAI()
MODEL = "gpt-5.4-mini"


def convert_timezone(date_time, from_timezone, to_timezone):
    """Convert a naive ISO datetime from one IANA timezone to another."""
    local_time = datetime.fromisoformat(date_time)
    source_time = local_time.replace(tzinfo=ZoneInfo(from_timezone))
    return source_time.astimezone(ZoneInfo(to_timezone)).isoformat()


# This schema tells the model when it can use the local Python function.
tools = [
    {
        "type": "function",
        "name": "convert_timezone",
        "description": "Convert an ISO datetime from one IANA timezone to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_time": {
                    "type": "string",
                    "description": "A local ISO datetime, for example 2026-01-20T14:30:00.",
                },
                "from_timezone": {
                    "type": "string",
                    "description": "The source IANA timezone, for example America/New_York.",
                },
                "to_timezone": {
                    "type": "string",
                    "description": "The target IANA timezone, for example Asia/Tokyo.",
                },
            },
            "required": ["date_time", "from_timezone", "to_timezone"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

user_request = (
    "What time is 2026-01-20T14:30:00 in America/New_York when converted "
    "to Asia/Tokyo? Use the conversion tool."
)

# First request: the model decides which function to call and supplies its arguments.
response = client.responses.create(model=MODEL, input=user_request, tools=tools)

tool_outputs = []
for item in response.output:
    if item.type == "function_call" and item.name == "convert_timezone":
        result = convert_timezone(**json.loads(item.arguments))
        tool_outputs.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"converted_time": result}),
            }
        )

# Second request gives the function result back to the model so it can answer clearly.
if tool_outputs:
    response = client.responses.create(
        model=MODEL,
        input=[*response.output, *tool_outputs],
        tools=tools,
    )

print(response.output_text)
