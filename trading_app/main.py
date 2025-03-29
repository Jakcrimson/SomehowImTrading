# main.py
# ... (keep all previous imports and class definition structure) ...
import random
import time
import sys
import math # Need for floor
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import threading
import queue

# ANSI escape codes (keep these)
COLOR_GREEN = "\033[92m"; COLOR_RED = "\033[91m"; COLOR_RESET = "\033[0m"
ARROW_UP = "▲"; ARROW_DOWN = "▼"

class TradingEnvironment:
    # ... (keep __init__, _update_prices, _calculate_stock_value, _record_state) ...
    # ... (keep display_status, plot_history) ...
    # ... (keep run_simulation_step) ...

    DEFAULT_UPDATE_INTERVAL_SECONDS = 15

    def __init__(self, resources, start_balance=1000.0, min_price=1.0,
                 base_drifts=None, base_volatilities=None,
                 shock_probability=0.02, shock_magnitude=0.15,
                 update_interval_seconds=None):
        self.resources = resources
        self.balance = start_balance
        self.inventory = {resource: 0 for resource in resources}
        self.min_price = min_price
        self.time_step = 0
        self.update_interval_seconds = update_interval_seconds if update_interval_seconds is not None else self.DEFAULT_UPDATE_INTERVAL_SECONDS
        if base_drifts is None: self.drifts = {"Wood": 0.0001, "Water": 0.00005, "Gold": 0.0003, "Oil": 0.0004, "Coffee": -0.0002}
        else: self.drifts = base_drifts
        if base_volatilities is None: self.volatilities = {"Wood": 0.015, "Water": 0.008, "Gold": 0.020, "Oil": 0.025, "Coffee": 0.018}
        else: self.volatilities = base_volatilities
        self.shock_probability = shock_probability
        self.shock_magnitude = shock_magnitude
        for r in self.resources:
            if r not in self.drifts: raise ValueError(f"Missing drift for resource: {r}")
            if r not in self.volatilities: raise ValueError(f"Missing volatility for resource: {r}")
        self.prices = {resource: round(random.uniform(10.0, 50.0), 2) for resource in resources}
        self.previous_prices = self.prices.copy()
        self.all_time_highs = self.prices.copy()
        self.all_time_lows = self.prices.copy()
        self.price_history = {resource: [] for resource in resources}
        self.balance_history = []
        self.stock_value_history = []
        self.total_wealth_history = []
        self._record_state()

    def _update_prices(self):
        self.previous_prices = self.prices.copy()
        dt = 1
        for resource in self.resources:
            drift = self.drifts[resource] * dt
            volatility = self.volatilities[resource]
            random_shock_gbm = np.random.normal(0, volatility) * math.sqrt(dt)
            percent_change_base = drift + random_shock_gbm
            if random.random() < self.shock_probability:
                shock_value = random.uniform(-self.shock_magnitude, self.shock_magnitude)
                percent_change_base += shock_value
            old_price = self.previous_prices[resource]
            new_price_raw = old_price * (1 + percent_change_base)
            new_price = round(max(self.min_price, new_price_raw), 2)
            self.prices[resource] = new_price
            if new_price > self.all_time_highs[resource]: self.all_time_highs[resource] = new_price
            if new_price < self.all_time_lows[resource]: self.all_time_lows[resource] = new_price

    def _calculate_stock_value(self):
        value = 0.0
        for r, q in self.inventory.items(): value += q * self.prices[r]
        return value

    def _record_state(self):
        for r in self.resources: self.price_history[r].append(self.prices[r])
        sv = self._calculate_stock_value(); tw = self.balance + sv
        self.balance_history.append(self.balance); self.stock_value_history.append(sv); self.total_wealth_history.append(tw)

    def display_status(self):
        stock_value = self._calculate_stock_value(); total_wealth = self.balance + stock_value
        print(f"\n{'='*15} Time Step: {self.time_step} (Prices updated at END of step) {'='*15}")
        financial_data = [["Available Cash", f"${self.balance:.2f}"], ["Current Stock Value", f"${stock_value:.2f}"], ["Total Wealth", f"${total_wealth:.2f}"]]
        print("\n-- Financials --"); print(tabulate(financial_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))
        inventory_data = []
        for r, q in self.inventory.items():
             if q > 0: inventory_data.append([r, q, f"${self.prices[r]:.2f}", f"${q * self.prices[r]:.2f}"])
        print("\n-- Inventory --");
        if not inventory_data: print("(Empty)")
        else: print(tabulate(inventory_data, headers=["Resource", "Quantity", "Unit Price", "Total Value"], tablefmt="fancy_grid", floatfmt=".2f"))
        market_data = []
        for resource, current_price in self.prices.items():
            previous_price = self.previous_prices.get(resource, current_price)
            change_str = "-"
            if self.time_step > 0 and previous_price is not None and previous_price != 0:
                 price_diff = current_price - previous_price; pct_change = (price_diff / previous_price) * 100
                 if abs(pct_change) < 0.001: change_str = f"  {pct_change:.2f}% "
                 elif pct_change > 0: change_str = f"{COLOR_GREEN}{ARROW_UP} {pct_change:.2f}%{COLOR_RESET}"
                 else: change_str = f"{COLOR_RED}{ARROW_DOWN} {abs(pct_change):.2f}%{COLOR_RESET}"
            elif self.time_step == 0: change_str = " - init - "
            ath = self.all_time_highs[resource]; atl = self.all_time_lows[resource]
            market_data.append([resource, f"${current_price:.2f}", change_str, f"${ath:.2f}", f"${atl:.2f}"])
        print("\n-- Market Prices --"); print(tabulate(market_data, headers=["Resource", "Price", "Change", "ATH", "ATL"], tablefmt="fancy_grid", disable_numparse=True))
        print("-" * 80); print(f"*** Enter actions below. Processed end of {self.update_interval_seconds}s interval. Use 'MAX' for quantity. Type 'QUIT' to exit. ***"); sys.stdout.flush()


    # --- execute_action is MODIFIED ---
    def execute_action(self, action_str):
        """Parses and executes ONE action string. Handles 'BUY/SELL <res> MAX'."""
        parts = action_str.strip().upper().split()
        if not parts:
            print(" [Skipping empty action]")
            return True

        command = parts[0]
        print(f" -> Processing: {action_str.strip()}...")

        if command == "QUIT":
            print("    QUIT command received. Exiting.")
            return False

        if command == "WAIT":
            print("    Action: WAIT (No trade executed).")
            return True

        if command in ["BUY", "SELL"]:
            # --- Input Validation ---
            if len(parts) != 3:
                print(f"    Error: Invalid {command} format. Use: {command} <RESOURCE> <QUANTITY|MAX>")
                return True

            _, resource_name_input, quantity_str = parts
            resource_name = next((r for r in self.resources if r.upper() == resource_name_input), None)

            if resource_name is None:
                print(f"    Error: Unknown resource '{resource_name_input}'.")
                return True

            # --- Determine Quantity ---
            quantity = 0
            is_max_action = (quantity_str == "MAX")

            # Use price from start of interval for calculations/execution
            price_used = self.previous_prices.get(resource_name, self.prices.get(resource_name, 0))
            if price_used <= 0:
                 print(f"    Error: Cannot {command} {resource_name} with invalid price ${price_used:.2f}.")
                 return True

            if command == "BUY":
                if is_max_action:
                    if self.balance <= 0: print(f"    Info: Insufficient balance (${self.balance:.2f}) to buy any {resource_name}."); return True
                    quantity = math.floor(self.balance / price_used)
                    if quantity <= 0: print(f"    Info: Insufficient balance (${self.balance:.2f}) to buy even one unit of {resource_name} at ${price_used:.2f}."); return True
                    print(f"    Calculating MAX BUY for {resource_name}: Balance ${self.balance:.2f} / Price ${price_used:.2f} = {quantity} units.")
                else: # Specific quantity
                    try: quantity = int(quantity_str); assert quantity > 0
                    except (ValueError, AssertionError): print(f"    Error: Invalid BUY quantity '{quantity_str}'."); return True

                # Execute Buy
                cost = price_used * quantity
                if cost > self.balance: print(f"    Error: Insufficient funds for {quantity} {resource_name} @ ${price_used:.2f}. Need ${cost:.2f}, have ${self.balance:.2f}"); return True
                self.balance -= cost; self.inventory[resource_name] += quantity
                print(f"    Action: Bought {quantity} {resource_name} @ ${price_used:.2f} for ${cost:.2f}. New balance: ${self.balance:.2f}")

            elif command == "SELL":
                if is_max_action:
                    quantity = self.inventory.get(resource_name, 0) # Get current holding
                    if quantity <= 0: print(f"    Info: No {resource_name} in inventory to SELL MAX."); return True
                    print(f"    Calculating MAX SELL for {resource_name}: Selling all {quantity} units.")
                else: # Specific quantity
                    try: quantity = int(quantity_str); assert quantity > 0
                    except (ValueError, AssertionError): print(f"    Error: Invalid SELL quantity '{quantity_str}'."); return True

                # Execute Sell
                if self.inventory.get(resource_name, 0) < quantity: print(f"    Error: Not enough {resource_name} to sell. Have {self.inventory.get(resource_name, 0)}, need {quantity}."); return True
                earnings = price_used * quantity
                self.balance += earnings; self.inventory[resource_name] -= quantity
                print(f"    Action: Sold {quantity} {resource_name} @ ${price_used:.2f} for ${earnings:.2f}. New balance: ${self.balance:.2f}")

            return True # Continue processing queue or finish step

        else: # Unknown command
            print(f"    Error: Unknown command '{command}'. Valid: BUY, SELL, WAIT, QUIT.")
            return True

    def run_simulation_step(self, action_queue):
        interval_duration = self.update_interval_seconds; start_time = time.monotonic()
        self.display_status()
        time.sleep(interval_duration)
        print(f"\n--- Interval End (Time Step {self.time_step}) ---"); print("Processing actions...")
        actions_processed = 0
        while not action_queue.empty():
            try:
                action_str = action_queue.get_nowait()
                actions_processed += 1
                should_continue = self.execute_action(action_str)
                if not should_continue: return False
            except queue.Empty: break
            except Exception as e: print(f"    Error processing action '{action_str}': {e}")
        if actions_processed == 0: print(" -> No actions entered. Defaulting to WAIT."); self.execute_action("WAIT")
        self._update_prices(); print("Prices updated for the next step.")
        self._record_state()
        self.time_step += 1
        return True

    def plot_history(self):
        if self.time_step <= 0: print("\nNo simulation steps completed..."); return
        time_steps = list(range(self.time_step + 1))
        plt.figure(figsize=(12, 6));
        for r, h in self.price_history.items():
            if len(h) == len(time_steps): plt.plot(time_steps, h, label=r, marker='.', ms=3, ls='-')
            else: print(f"Warn: History length mismatch for {r}.")
        plt.xlabel("Time Step"); plt.ylabel("Price ($)"); plt.title("Resource Price History (GBM Sim)"); plt.legend(); plt.grid(True); plt.tight_layout();
        plt.figure(figsize=(12, 6));
        if len(self.balance_history)==len(time_steps): plt.plot(time_steps, self.balance_history, label="Cash", marker='.')
        if len(self.stock_value_history)==len(time_steps): plt.plot(time_steps, self.stock_value_history, label="Stock Value", marker='.')
        if len(self.total_wealth_history)==len(time_steps): plt.plot(time_steps, self.total_wealth_history, label="Total Wealth", marker='.', lw=2)
        plt.xlabel("Time Step"); plt.ylabel("Value ($)"); plt.title("Player Financial History"); plt.legend(); plt.grid(True); plt.tight_layout();
        print("\nDisplaying plots... Close plot windows to exit."); plt.show()


# --- Input Thread Function (Unchanged) ---
def input_listener(action_queue, stop_event):
    print("[Input thread started. Type commands anytime.]")
    while not stop_event.is_set():
        try:
            command = input()
            if stop_event.is_set(): break
            if command: action_queue.put(command.strip())
        except EOFError: print("[Input thread detected EOF. Exiting.]"); break
        except Exception as e: print(f"[Input thread error: {e}]"); time.sleep(0.5)


# --- Main Execution (Unchanged) ---
if __name__ == "__main__":
    import numpy as np
    RESOURCES = ["Wood", "Water", "Gold", "Oil", "Coffee"]
    INTERVAL = 15
    command_queue = queue.Queue()
    shutdown_event = threading.Event()
    env = TradingEnvironment(resources=RESOURCES, start_balance=1000.0, min_price=0.50,
                             shock_probability=0.03, shock_magnitude=0.20, update_interval_seconds=INTERVAL)
    input_thread = threading.Thread(target=input_listener, args=(command_queue, shutdown_event), daemon=True)
    input_thread.start()
    print("Welcome to the Asynchronous Trading Simulation!")
    print(f"Market updates and actions resolve every {INTERVAL} seconds.")
    print("Market display includes Price Change%, ATH, ATL.")
    print("Use 'BUY <RES> MAX' or 'SELL <RES> MAX'.")
    running = True
    try:
        while running:
            running = env.run_simulation_step(command_queue)
    except KeyboardInterrupt: print("\nCtrl+C detected. Shutting down."); running = False
    finally:
        print("\nSignaling input thread to stop..."); shutdown_event.set()
        print("\n--- Simulation Ended ---"); env.display_status(); env.plot_history(); print("\nExiting program.")