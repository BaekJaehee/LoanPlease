package com.d105.loanplease.domain.store.adapter.out;

import com.d105.loanplease.domain.store.domain.Item;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ItemRepository extends JpaRepository<Item, Long> {
}
