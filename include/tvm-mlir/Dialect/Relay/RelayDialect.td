#ifndef RELAYDIALECT_TD
#define RELAYDIALECT_TD

include "mlir/IR/OpBase.td"

def Relay_Dialect : Dialect {
  let name = "relay";
  let cppNamespace = "::relay";
}

#endif
