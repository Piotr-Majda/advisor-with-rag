# AI Agent Flow Example

## Example Query

```
User: "Compare Bitcoin's current price with its price from 6 months ago, and analyze any relevant news that might explain the changes. Also check if there are any important documents in our repository about cryptocurrency investment strategies."
```

## Process Flow

### 1. Initial Query Analysis

```python
# Agent receives the query and analyzes required tools
analysis_prompt = """
Analyze the following user query and determine which tools would be most helpful.
Query: "Compare Bitcoin's current price with its price from 6 months ago, and analyze any relevant news that might explain the changes. Also check if there are any important documents in our repository about cryptocurrency investment strategies."

Available tools:
- web_search: Search the web for real-time information and news
- document_search: Search through internal document repository
- market_data: Get historical and current cryptocurrency prices
- calculator: Perform numerical calculations

Respond with a JSON array of tool names, ordered by relevance.
"""

# AI Model Response
{
    "tools": [
        "market_data",
        "web_search",
        "document_search",
        "calculator"
    ]
}
```

### 2. Tool Execution Sequence

#### Step 1: Market Data Query

```python
# First tool execution: market_data
tool_call_1 = {
    "name": "market_data",
    "arguments": {
        "symbol": "BTC",
        "current": True,
        "historical": True,
        "months_ago": 6
    }
}

# Tool Response
{
    "status": "success",
    "data": {
        "current_price": 66789.23,
        "historical_price": 27891.45,
        "change_percentage": 139.46
    }
}
```

#### Step 2: News Analysis

```python
# Second tool execution: web_search
tool_call_2 = {
    "name": "web_search",
    "arguments": {
        "query": "Bitcoin price increase factors last 6 months ETF approval institutional adoption"
    }
}

# Tool Response
{
    "status": "success",
    "data": [
        {
            "title": "Bitcoin ETF Approval Impact",
            "summary": "SEC approval of spot Bitcoin ETFs led to significant price increases..."
        },
        {
            "title": "Institutional Adoption Growth",
            "summary": "Major financial institutions increased Bitcoin holdings..."
        }
    ]
}
```

#### Step 3: Document Search

```python
# Third tool execution: document_search
tool_call_3 = {
    "name": "document_search",
    "arguments": {
        "query": "cryptocurrency investment strategies Bitcoin analysis"
    }
}

# Tool Response
{
    "status": "success",
    "data": [
        {
            "document_id": "inv_strat_001",
            "title": "Crypto Investment Framework 2024",
            "relevance_score": 0.89,
            "excerpt": "..."
        }
    ]
}
```

#### Step 4: Calculate Returns

```python
# Fourth tool execution: calculator
tool_call_4 = {
    "name": "calculator",
    "arguments": {
        "operation": "roi",
        "initial_value": 27891.45,
        "final_value": 66789.23
    }
}

# Tool Response
{
    "status": "success",
    "data": {
        "roi": 139.46,
        "formatted": "139.46%"
    }
}
```

### 3. AI Processing & Response Generation

```python
# AI combines all tool results and generates response
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query},
    {"role": "tool", "content": str(market_data_response)},
    {"role": "tool", "content": str(web_search_response)},
    {"role": "tool", "content": str(document_search_response)},
    {"role": "tool", "content": str(calculator_response)}
]

response = await openai.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=messages,
    stream=True
)
```

### 4. Final Response to User

```
Based on my analysis:

1. Price Comparison:
   - Current Bitcoin price: $66,789.23
   - Price 6 months ago: $27,891.45
   - Total increase: 139.46%

2. Key Factors for Price Change:
   - SEC approval of spot Bitcoin ETFs was a major catalyst
   - Increased institutional adoption, with major financial institutions adding Bitcoin to their portfolios

3. Internal Documentation:
   - Found relevant document: "Crypto Investment Framework 2024"
   - This document contains updated investment strategies aligned with current market conditions

Would you like me to:
1. Provide more detailed news analysis?
2. Extract specific strategies from our investment framework?
3. Calculate potential future projections based on current trends?
```

## Implementation Details

### 1. Tool Selection Logic

```python
class ToolSelector:
    async def select_tools(self, query: str) -> List[BaseTool]:
        # 1. Analyze query intent
        intents = await self._analyze_query_intent(query)

        # 2. Match intents with tool capabilities
        relevant_tools = []
        for intent in intents:
            matching_tools = self._match_intent_to_tools(intent)
            relevant_tools.extend(matching_tools)

        # 3. Order by relevance and remove duplicates
        return self._prioritize_tools(relevant_tools)
```

### 2. Memory Management

```python
class AgentMemory:
    async def add_tool_result(self, tool_name: str, result: MCPResponse):
        # Store tool execution result
        self.tool_results.append({
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.now()
        })

        # Update tool success statistics
        self._update_tool_stats(tool_name, result.status == "success")
```

### 3. Error Recovery

```python
async def _handle_tool_failure(self, tool_name: str, error: Exception):
    # Log failure
    await self.memory.log_tool_failure(tool_name, error)

    # Try alternative tool if available
    alternative_tool = await self.tool_selector.get_alternative(tool_name)
    if alternative_tool:
        return await self._execute_with_retry(alternative_tool)

    # Return graceful failure response
    return MCPResponse(
        status="error",
        error=f"Tool execution failed and no alternative available: {str(error)}"
    )
```

## Benefits of This Approach

1. **Intelligent Tool Selection**

   - Context-aware tool choice
   - Dynamic prioritization
   - Failure recovery

2. **Efficient Resource Usage**

   - Tools called only when needed
   - Results cached in memory
   - Parallel execution when possible

3. **Natural Interaction**

   - Coherent responses
   - Context maintenance
   - Progressive disclosure

4. **Reliability**
   - Error handling
   - Retry mechanisms
   - Alternative tool selection
