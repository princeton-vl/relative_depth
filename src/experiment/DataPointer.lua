require 'xlua'
local DataPointer = torch.class('DataPointer')

function DataPointer:__init(n_total)
    self.n_total = n_total
    if self.n_total > 0 then
        self.idx_perm = torch.randperm(self.n_total)
        self.current_pos = 1
    else
        self.idx_perm = nil
        self.current_pos = nil
    end
end


function DataPointer:load_next_batch(batch_size)
    if self.n_total <= 0 then
        return nil
    end
    
    if batch_size == 0 then
        return nil
    end

    -- get indices
    local indices = torch.Tensor()
    if batch_size + self.current_pos - 1 <= self.n_total then        
        indices = self.idx_perm:narrow(1, self.current_pos, batch_size)
    else        
        local rest = batch_size + self.current_pos - 1 - self.n_total

        local part1 = self.idx_perm:narrow(1, self.current_pos, (self.n_total - self.current_pos + 1) )
        local part2 = self.idx_perm:narrow(1, 1, rest)
        indices = torch.cat(part1, part2)
    end


    -- update pointer
    self.current_pos = self.current_pos + batch_size
    if self.current_pos >= self.n_total then
        -- reset to the initial position
        self.current_pos = 1

        -- reshuffle the images
        self.idx_perm = torch.randperm(self.n_total);
    end

    return indices
end

