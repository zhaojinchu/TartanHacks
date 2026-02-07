import type { BinType } from "../types";

const BIN_NAME_BY_ID: Record<string, string> = {
  "bottles-cans": "Bottles / Cans",
  compostables: "Compostables",
  landfill: "Landfill"
};

const BIN_NAME_BY_TYPE: Record<BinType, string> = {
  recycle: "Bottles / Cans",
  compost: "Compostables",
  landfill: "Landfill"
};

function titleCase(input: string): string {
  return input
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}

export function prettyBinNameFromId(binId: string): string {
  const mapped = BIN_NAME_BY_ID[binId];
  if (mapped) {
    return mapped;
  }

  return titleCase(binId.replace(/[_-]+/g, " "));
}

export function prettyBinNameFromType(binType: BinType): string {
  return BIN_NAME_BY_TYPE[binType] ?? titleCase(binType.replace(/[_-]+/g, " "));
}

export function prettyGroupKey(value: string): string {
  if (value in BIN_NAME_BY_ID) {
    return prettyBinNameFromId(value);
  }

  if (value in BIN_NAME_BY_TYPE) {
    return prettyBinNameFromType(value as BinType);
  }

  return titleCase(value.replace(/[_-]+/g, " "));
}

export function formatDateTimeMinutes(value: string | null | undefined): string {
  if (!value) {
    return "N/A";
  }

  return new Date(value).toLocaleString([], {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  });
}
