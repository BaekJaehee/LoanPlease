package com.d105.loanplease.store;

import com.d105.loanplease.domain.store.application.service.ItemService;
import com.d105.loanplease.domain.store.domain.Item;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

@SpringBootTest
public class ItemServiceTest {

    @Autowired
    private ItemService itemService;

    @Test
    void 아이템전체조회() {
        List<Item> items = itemService.inquiryAllItems();
        assertThat(items.size()).isEqualTo(1);
    }

    @Test
    void 특정아이템조회() {
        Long itemId = 1L;
        Item item = itemService.inquiryItemById(itemId);

        assertThat(item.getName()).isEqualTo("게임 시간 증가+1분");
        assertThat(item.getContent()).isEqualTo("해당 아이템은 게임 시간을 1분 증가시켜줍니다.");
        assertThat(item.getPrice()).isEqualTo(1500);
    }

    @Test
    void 아이템구매() {

    }

}
