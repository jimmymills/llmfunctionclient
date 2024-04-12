import inspect 
import re
import json
import enum
from typing import Callable, Optional

def parse_description(func: Callable) -> tuple[str, dict[str, str]]:
  """
  Parses the description of a function and returns the top level function description and parameter descriptions.
  """
  s = func.__doc__
  if not s:
     return '', {}
  # Split the string into lines
  lines = s.strip().split('\n')

  # The first line is the top description
  top_description = lines[0].strip()

  # The rest of the lines are parameter descriptions
  param_descriptions = {}
  for line in lines[1:]:
      # Use a regular expression to split the line into the parameter name and description
      match = re.match(r'\s*(\w+):\s*(.*)', line)
      if match:
          param_name, param_description = match.groups()
          param_descriptions[param_name] = param_description.strip()

  return top_description, param_descriptions

def map_type(param: inspect.Parameter) -> str:
  """
  Maps a parameter type to a JSON Schema type

  Raises a ValueError if the parameter type is not supported
  """
  if issubclass(param.annotation, str):
      return 'string'
  elif issubclass(param.annotation, int):
      return 'integer'
  else:
      raise ValueError(f'Unsupported parameter type: {param.annotation}')

def map_enum(param: inspect.Parameter) -> list[str] | None:
  """
  Gets the enum argument for a parameter if it is an enum, otherwise returns None
  """
  if issubclass(param.annotation, enum.Enum):
    return [e.name for e in param.annotation]
  return None

def parse_parameters(func: Callable, param_descriptions: dict[str, str]) -> dict[str, dict[str, str] | list[str] | str]:
  """
  Parses the parameters of a function and returns a JSON Schema object
  """
  parameters = {}
  for name, param in inspect.signature(func).parameters.items():
      param_type = map_type(param)
      options = map_enum(param)
      parameters[name] = {
          'type': param_type,
      }
      if options:
          parameters[name]['enum'] = options
      if name in param_descriptions:
          parameters[name]['description'] = param_descriptions[name]
  return parameters

def get_required(func: Callable) -> list[str]:
  """
  Gets the required parameters for a function, checking which don't have default values
  """
  return [name for name, param in inspect.signature(func).parameters.items() if param.default == inspect.Parameter.empty]

def to_tool(func: Callable) -> dict[str, dict[str, str] | dict[str, dict[str, str]] | list[str] | str]:
  """
  Converts a function to a tool object for use with an OpenAI-like API.

  The function must contain type annotations and only have parameters that are strings, 
  integers and can be enums of those two types. Optionally, including a docstring for the function
  will be used as the description for the tool for the first line. Subsequent lines will be used as
  descriptions for the parameters with the format <parameter_name>: <parameter_description>
  """
  top_description, param_descriptions = parse_description(func)
  tool = {
    'type': 'function',
    'function': {
      'name': func.__name__,
      'parameters': {
        'type': 'object',
        'properties': parse_parameters(func, param_descriptions),
        'required': get_required(func), 
      },
    }
  }
  if top_description:
    tool['description'] = top_description
  return tool

class FunctionClient:
  """
  A client for interacting with a language model that supports function calls.
  """
  def __init__(self, client, model: str, functions: list[Callable], messages: Optional[list[dict[str, str]]]=None):
    """
    Initializes the client with a language model, functions, and messages.

    client: The OpenAI-like client
    model: The language model to use
    functions: A default list of functions to call
    messages: A list of messages to start the conversation with
    """
    self.messages = messages or []
    self.model = model
    self.client = client
    self.functions = functions

  @staticmethod
  def funcs_to_tools(funcs: list[Callable]) -> tuple[list[dict[str, dict[str, str] | dict[str, dict[str, str]] | list[str] | str]], dict[str, Callable]]:
    """
    Converts a list of functions to a list of tools and a dictionary mapping function names to functions.
    """
    return [to_tool(func) for func in funcs], {func.__name__: func for func in funcs}

  def add_message(self, content: str, role: Optional[str]='user'):
    """
    Adds a message to the conversation.
    """
    self.messages.append({'role': role, 'content': content})

  def __send_message(self, functions: list[Callable], force_function:Optional[str | Callable]=None, model: Optional[str]=None):
    """
    Sends a message to the language model and processes the response.
    """
    tools, tools_map = self.funcs_to_tools(functions)
    args = {
      'model': model,
      'messages': self.messages,
      'tools': tools,
    }
    if force_function:
       args['tool_choice'] = {"type": "function", "function": {"name": force_function}}
    response = self.client.chat.completions.create(
      model=self.model,
      messages=self.messages,
      tools=tools,
    )
    message = response.choices[0].message
    if message.tool_calls:
      for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = tools_map[tool_call.function.name](**args)
        self.messages.append({"role": "function", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
      return False
    else:
      self.messages.append({"role": "assistant", "content": message.content})
      return True

  def send_message(self, content: Optional[str]=None, role: Optional[str]=None, model: Optional[str]=None, functions: Optional[list[Callable]]=None, force_function: Optional[str | Callable]=None, num_calls: Optional[int]=100) -> str:
    """
    Sends a message to the language model and processes the response, responding to tool call requests until a text response is received.
    Can be called without a content argument to just use current messages.

    content: Optional, the content of the message.
    role: Optional, the role of the message, defaults to 'user'.
    functions: Optional, the list of functions to use, defaults to the list of functions passed to the constructor.
    force_function: Optional, a function or the name of the function to force the model to call.
    """
    if content:
      self.add_message(content, role)
    if not functions:
       functions = self.functions
    if callable(force_function):
       force_function = force_function.__name__
    if not model:
        model = self.model
    done = False
    i = 0
    # Only force_function for the first call to avoid tool call loops
    done = self.__send_message(functions, force_function, model=model)
    i += 1
    while not done and i < num_calls:
      done = self.__send_message(functions, model=model)
    return self.messages[-1]['content']
