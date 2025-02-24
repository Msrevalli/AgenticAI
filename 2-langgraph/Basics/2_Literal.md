# **📌 Understanding `Literal` in Python Typing**
In Python, **`Literal`** is a **type hint** that restricts a variable or function return value to **specific predefined values**.

---

## **🚀 Why Use `Literal`?**
✔ **Restricts Function Outputs** – Ensures only **expected values** are returned.  
✔ **Improves Type Safety** – Helps **prevent invalid assignments**.  
✔ **Enhances Code Readability** – Clearly defines **possible options**.  
✔ **Works with Type Checkers** – Helps **mypy, Pylance, Pyright detect issues**.

---

## **📌 Syntax of `Literal`**
```python
from typing import Literal

def get_status() -> Literal["success", "failure"]:
    return "success"  # ✅ Valid
```
✔ **`get_status()` can only return `"success"` or `"failure"`.**  
❌ **Returning `"error"` would cause a type error.**

---

## **📌 Example: Using `Literal` in Function Parameters**
```python
def set_mode(mode: Literal["auto", "manual", "hybrid"]):
    print(f"Mode set to: {mode}")

set_mode("auto")   # ✅ Valid
set_mode("hybrid") # ✅ Valid
set_mode("test")   # ❌ Type Error: "test" is not a valid Literal value
```
✔ **Only `"auto"`, `"manual"`, or `"hybrid"` are allowed!**

---

## **📌 Example: Using `Literal` in Class Attributes**
```python
from typing import TypedDict, Literal

class Car(TypedDict):
    model: str
    transmission: Literal["automatic", "manual"]

car: Car = {"model": "Tesla", "transmission": "automatic"}  # ✅ Valid
car_invalid: Car = {"model": "Ford", "transmission": "semi-auto"}  # ❌ Type Error
```
✔ **Ensures `transmission` only has `"automatic"` or `"manual"` values.**

---

## **📌 Example: Using `Literal` in LangGraph Decision Nodes**
```python
from typing import Literal

def choose_route(state) -> Literal["second_node", "third_node"]:
    if state["graph_state"] == "AI":
        return "second_node"
    return "third_node"

print(choose_route({"graph_state": "AI"}))  # ✅ "second_node"
print(choose_route({"graph_state": "ML"}))  # ✅ "third_node"
print(choose_route({"graph_state": "unknown"}))  # ✅ Still returns a valid option
```
✔ **AI workflow follows only `"second_node"` or `"third_node"`.**

---

## **📌 When to Use `Literal`?**
| **Scenario** | **Why Use `Literal`?** |
|-------------|------------------|
| **Restricting Function Outputs** | Ensures function returns only valid values. |
| **Ensuring Valid Parameters** | Prevents passing unexpected values to a function. |
| **Decision-Making in AI** | AI selects only from predefined paths. |
| **Type-Safe Configuration Settings** | Defines strict config values like `"debug"`, `"production"`. |

---

## **🔥 Final Thoughts**
🚀 **`Literal` is a powerful typing tool** that ensures **code correctness, improves readability, and prevents invalid values**.  
🚀 **Best used for AI workflows, API modes, decision-making, and configuration settings.**  
