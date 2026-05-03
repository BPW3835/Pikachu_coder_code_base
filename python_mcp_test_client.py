# import asyncio
# from fastmcp import Client

# async def main():
#     # Point the client at your server file
#     client = Client("mcp_server.py")

#     # Connect to the server
#     async with client:
#         # List available tools
#         tools = await client.list_tools()
#         print("Available tools:")
#         for tool in tools:
#             print(f"  - {tool.name}: {tool.description}")

#         print("\n" + "=" * 50 + "\n")

#         # Call the weather tool
#         result = await client.call_tool(
#             "get_weather",
#             {"city": "Tokyo"}
#         )
#         print(f"Weather result: {result}")

# if __name__ == "__main__":
#     asyncio.run(main())


import os
import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main():
    # Collect env vars needed by the server
    env = {}
    if "OPENWEATHER_API_KEY" in os.environ:
        env["OPENWEATHER_API_KEY"] = os.environ["OPENWEATHER_API_KEY"]

    transport = StdioTransport(
        sys.executable,              # use current Python
        ["mcp_server.py"],      # or your actual filename
        env=env,                     # pass env into the server process
    )

    client = Client(transport)

    async with client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")

        print("\n" + "=" * 50 + "\n")

        result = await client.call_tool(
            "get_weather",
            {"city": "Tokyo"},
        )

        print("Weather result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())