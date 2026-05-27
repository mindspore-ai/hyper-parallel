# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""func tracer module"""
import ast
import sys
import inspect
import textwrap
import linecache


class _FuncTracer:
    """Func tracer class"""

    def __init__(self):
        self.ind = 0
        self.trace_str = ""
        self.code_trees = {}
        self.max_sub_length = 30

    def add2trace(self, s):
        """log"""
        self.trace_str += "\t" * self.ind + s + "\n"

    def is_constant_expr(self, s):
        """whether it's a leave node"""
        node = ast.parse(s, mode="eval").body
        return isinstance(node, (ast.Constant, ast.Attribute, ast.Name))

    def scrap_term_symbols(self, s):
        """get all leave nodes from source code"""
        res = []
        tree = ast.parse(s)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if not any(r.startswith(node.id) for r in res):
                    res += [node.id]
            elif isinstance(node, ast.Constant):
                res += [str(node.value)]
            elif isinstance(node, ast.Attribute):
                # Reconstruct full attribute chain like obj.attr1.attr2
                value = node
                attr_chain = []
                while isinstance(value, ast.Attribute):
                    attr_chain.insert(0, value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    attr_chain.insert(0, value.id)
                res += [".".join(attr_chain)]
        res = sorted(res, key=len, reverse=True)
        return res

    def substitute(self, s, local):
        """substitute leave nodes by their numerical values"""
        nodes = self.scrap_term_symbols(s)
        for n in nodes:
            attr_chain = n.split(".")
            if attr_chain[0] in local:
                if len(attr_chain) == 1:
                    target = str(local[n])
                    if len(target) < self.max_sub_length:
                        s = s.replace(n, target)
                    else:
                        s = s.replace(n, type(local[n]).__name__)
                else:
                    attr = local[attr_chain[0]]
                    for a in attr_chain[1:]:
                        attr = getattr(attr, a)
                    target = str(attr)
                    if len(target) < self.max_sub_length:
                        s = s.replace(n, target)
                    else:
                        s = s.replace(n, type(attr).__name__)
        return s

    def fetch_node_from_lineno(self, lineno, co):
        """Get AST node from a line num in source code"""
        for node in ast.walk(self.code_trees[co]):
            if hasattr(node, "lineno") and node.lineno == lineno:
                return node
        return None

    def line_tracer(self, frame, event, _):
        """Trace executed code line"""
        co = frame.f_code
        func_name = co.co_name

        # Handling function call
        if event == "call":
            self.extract_ast(co)
            _, _, _, values = inspect.getargvalues(frame)
            args_str = ",".join(
                f"{k}={v if isinstance(v, (int, float)) else type(v).__name__}"
                for k, v in values.items()
            )
            self.add2trace(f"::{func_name}({args_str})")
            self.ind += 1

        # Handling instruction
        elif event == "line":
            line = linecache.getline(co.co_filename, frame.f_lineno).strip()
            ignored_statements = ("if ", "elif ", "else ", "for ", "while ")
            op_equals = ["*=", "+=", "-=", "/=", "%=", "="]
            # Only capture assignments and returns
            if not line.startswith(ignored_statements):
                base_str = f"{func_name}->"
                if "=" in line:
                    try:
                        ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        node = self.fetch_node_from_lineno(frame.f_lineno, co)
                        if not node:
                            return self.line_tracer
                        line = ast.unparse(node)
                    sign = next(o for o in op_equals if o in line)
                    left, right = line.split(sign, 1)
                    left, right = left.strip(), right.strip()
                    if sign != "=":
                        right = f"{left} {sign[0]} {right}"
                    self.add2trace(f"{base_str} {left} = {right}")
                    if not self.is_constant_expr(right):
                        val = eval(  # pylint: disable=eval-used
                            right, frame.f_globals, frame.f_locals
                        )
                        sub_right = self.substitute(right, frame.f_locals)
                        base_str = f"{base_str} {left}"
                        if not self.is_constant_expr(sub_right):
                            self.add2trace(
                                f"{len(base_str)*' '} = {sub_right}"
                            )
                            self.add2trace(f"{len(base_str)*' '} = {val}")
                        else:
                            self.add2trace(f"({right}) = {val}")
                # Handle return statements
                elif line.startswith("return"):
                    lreturn = len("return")
                    expr = line[lreturn:].strip()
                    try:
                        ast.literal_eval(expr)
                    except (ValueError, SyntaxError):
                        node = self.fetch_node_from_lineno(frame.f_lineno, co)
                        if not node:
                            return self.line_tracer
                        expr = ast.unparse(node)[lreturn:].strip()
                    val = eval(  # pylint: disable=eval-used
                        expr, frame.f_globals, frame.f_locals
                    )
                    sub_expr = self.substitute(expr, frame.f_locals)
                    self.add2trace(f"{base_str} return {expr}")
                    if not self.is_constant_expr(sub_expr):
                        self.add2trace(f"{len(base_str)*' '} = {sub_expr}")
                        self.add2trace(f"{len(base_str)*' '} = {val}")
                    else:
                        self.add2trace(f"{len(base_str)*' '} = {val}")

        elif event == "return":
            self.ind -= 1

        return self.line_tracer

    def extract_ast(self, fun):
        """Get AST from function source code"""
        if fun not in self.code_trees:
            self.code_trees[fun] = ast.parse(
                textwrap.dedent(inspect.getsource(fun))
            )
            _, start_line = inspect.getsourcelines(fun)
            # Extract file's real line numbers for AST
            for node in ast.walk(self.code_trees[fun]):
                if hasattr(node, "lineno"):
                    node.lineno = start_line + node.lineno - 1
                    node.end_lineno = start_line + node.end_lineno - 1

    def wrap(self, fun):
        """Wrapper"""

        def tracked_fun(*args, **kwargs):
            """Wrapper"""
            sys.settrace(self.line_tracer)
            try:
                print("Tracing...")
                res = fun(*args, **kwargs)
                print(self.trace_str)
                self.trace_str = ""
                return res
            finally:
                sys.settrace(None)
            return res

        return tracked_fun
