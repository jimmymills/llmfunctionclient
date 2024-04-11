# Python LLM Function Client

The purpose of this library is to simplify using function calling with OpenAI-like API clients. Traditionally, you would have to rewrite your functions into JSON Schema and write logic to handle tool calls in responses. With this library, you can convert python functions into JSON schema by simply calling `to_tool(func)` or you can create a client that will handle those tool calls for you and simply pass back a response once the tool call chain is finished by creating an instance of `FunctionClient`.

## Installation

To install simply run:
`pip install llmfunctionclient`

## Requirements for Functions

Functions used with this library must have type annotations for each parameter. You do not have to have an annotation for the return type of the function.
Currently, the supported types are string, int, StrEnum and IntEnum.
If the type is a StrEnum or IntEnum, the valid values will be included as part of the function tool spec.

Optionally, you can include a docstring to add descriptions. The first line of the docstring will be considered the description of the function. Subsequent lines should be of the format `<parameter_name>: <description>`

For example:
```python
def get_weather(location: str):
  """
  Gets the weather

  location: where to get the forecast for
  """
  return f"The weather in {location} is 75 degrees"
```

This function will have "Gets the weather" as the function description and the location parameter will have the description "where to get the forecast for"

## FunctionClient

The `FunctionClient` class is made to abstract away the logic of passing along tool calls by taking in a list of functions that are allowed to be called by the LLM client, running any tool calls required by LLM client responses until it is left with just text to respond with.

```python
from llmfunctionclient import FunctionClient
from openai import OpenAI

def get_weather(location: str):
  """
  Gets the weather

  location: where to get the forecast for
  """
  return f"The weather in {location} is 75 degrees"

client = FunctionClient(OpenAI(), "gpt-3.5-turbo", [get_weather])
client.add_message("You are a helpful weather assistant.", "system")
response = client.send_message("What's the weather in LA?", "user")
print(response) # "The current weather in Los Angeles is 75 degrees"
```

When this is run, the following happens under the hood:  
1. The two messages specified here will be submitted to the LLM Client
2. The LLM Client responds with a tool called for "get_weather"
3. The get_weather function is called and the result is appended as a message
4. The LLM Client is called again with the function result.
5. The LLM Client Responds with an informed answer.
6. This response text is passed back.

You can pass functions into the constructor of the client to create the default set of tools for every message as well as pass in the `functions` kwarg to `send_message` to specify a specific set of functions for that portion of the conversation.

To force the LLM to use a specific function, you can pass the `force_function` kwarg with the function (or its name) you want the LLM to use and it will be provided as the tool_choice parameter for the chat completion endpoint.

## to_tool

If you want to continue using any other LLM clients and just want the ability to convert python functions into JSON Schema compatible with the function calling spec, you can simply import the function to_tool and call that on the function.

Example:
```python
def get_weather(location: str):
  """
  Gets the weather

  location: where to get the forecast for
  """
  return f"The weather in {location} is 75 degrees"

```

Calling to_tool(get_weather) returns the following object

```python
{'type': 'function',
 'function': {'name': 'get_weather',
  'parameters': {'type': 'object',
   'properties': {'location': {'type': 'string',
     'description': 'where to get the forecast for'}},
   'required': ['location']}},
 'description': 'Gets the weather'}
```

This can then be used with the normal OpenAI client like this:
```python
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=messages,
  tools=[to_tool(get_weather)],
  tool_choice="auto"
)
```
