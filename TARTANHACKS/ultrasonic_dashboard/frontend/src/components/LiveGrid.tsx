import { useMemo, useState } from "react";

import type { BinMeasurement } from "../types";
import { BinCard } from "./BinCard";

interface Props {
  bins: BinMeasurement[];
  onEmpty: (binId: string) => void;
}

export function LiveGrid({ bins, onEmpty }: Props): JSX.Element {
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [locationFilter, setLocationFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  const locations = useMemo(() => Array.from(new Set(bins.map((bin) => bin.location))).sort(), [bins]);

  const filtered = useMemo(
    () =>
      bins.filter((bin) => {
        if (typeFilter !== "all" && bin.bin_type !== typeFilter) {
          return false;
        }
        if (locationFilter !== "all" && bin.location !== locationFilter) {
          return false;
        }
        if (statusFilter !== "all" && bin.status !== statusFilter) {
          return false;
        }
        return true;
      }),
    [bins, typeFilter, locationFilter, statusFilter]
  );

  return (
    <section>
      <div className="filters">
        <label>
          Type
          <select value={typeFilter} onChange={(event) => setTypeFilter(event.target.value)}>
            <option value="all">All</option>
            <option value="recycle">Bottles / Cans</option>
            <option value="compost">Compostables</option>
            <option value="landfill">Landfill</option>
          </select>
        </label>

        <label>
          Location
          <select value={locationFilter} onChange={(event) => setLocationFilter(event.target.value)}>
            <option value="all">All</option>
            {locations.map((location) => (
              <option value={location} key={location}>
                {location}
              </option>
            ))}
          </select>
        </label>

        <label>
          Status
          <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
            <option value="all">All</option>
            <option value="normal">Normal</option>
            <option value="almost_full">Almost Full</option>
            <option value="full">Full</option>
            <option value="sensor_error">Sensor Error</option>
          </select>
        </label>
      </div>

      <div className="card-grid">
        {filtered.map((bin) => (
          <BinCard key={bin.bin_id} bin={bin} onEmpty={onEmpty} />
        ))}
      </div>
    </section>
  );
}
