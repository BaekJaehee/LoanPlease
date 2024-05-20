package com.d105.loanplease.domain.store.adapter.persistence;

import com.d105.loanplease.domain.store.adapter.out.ItemRepository;
import com.d105.loanplease.domain.store.application.port.out.ItemPort;
import com.d105.loanplease.domain.store.domain.Item;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class ItemPersistenceAdapter implements ItemPort {

    @Autowired
    private ItemRepository itemRepository;

    @Override
    public List<Item> findAll() {
        return itemRepository.findAll();
    }

    @Override
    public Item findById(final Long id) {
        return itemRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("해당 게임 아이템이 존재하지 않습니다."));
    }
}
