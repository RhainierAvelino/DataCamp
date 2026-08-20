tools = [
    {
        # Define a function tool called convert_timezone
        "type": "function",
        "name": "convert_timezone",
        "description": "Convert a datetime from one timezone to another using the OpenTimezone API.",
        "parameters": {
            "type": "object",
            # Define the parameter names, types, and descriptions
            "properties": {
                "date_time": {
                    "type": "string",
                    "description": "The datetime string in ISO format (e.g., '2025-01-20T14:30:00')"
                },
                "from_timezone": {
                    "type": "string",
                    "description": "The source timezone (e.g., 'America/New_York', 'Asia/Tokyo')"
                },
                "to_timezone": {
                    "type": "string",
                    "description": "The target timezone (e.g., 'Europe/London', 'Australia/Sydney')"
                }
            },
            # Ensure that all three parameters are required
            "required": ["date_time", "from_timezone", "to_timezone"],
            "additionalProperties": False
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "What time is 2:30pm on January 20th in New York in Tokyo time?"
    }
]

response = client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=tools
)

messages += response.output

# Process function calls and execute the timezone conversion
for item in response.output:
    if item.type == "function_call":
        if item.name == "convert_timezone":
            timezone_result = convert_timezone(
                **json.loads(item.arguments)
            )

            # Append function output to messages
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(
                        {"convert_timezone": timezone_result}
                    )
                }
            )

# Make second API request with function results
response = client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=tools
)

print(response.output_text)