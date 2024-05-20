package com.d105.loanplease.domain.store.application.port.out;

import com.d105.loanplease.domain.store.domain.Item;

import java.util.List;

public interface ItemPort {

    List<Item> findAll();
    Item findById(Long id);
}
