package io.leavesfly.tinyai.minimind.api;

import java.util.*;

/**
 * 轻量级JSON工具类
 * 
 * 零依赖的简化JSON解析和生成,仅支持API所需的基础功能
 * 
 * @author leavesfly
 * @since 2024
 */
public class SimpleJSON {
    
    /**
     * 将对象转换为JSON字符串
     */
    public static String toJSON(Object obj) {
        if (obj == null) {
            return "null";
        }
        
        if (obj instanceof String) {
            return "\"" + escape((String) obj) + "\"";
        }
        
        if (obj instanceof Number || obj instanceof Boolean) {
            return obj.toString();
        }
        
        if (obj instanceof Map) {
            return mapToJSON((Map<?, ?>) obj);
        }
        
        if (obj instanceof List) {
            return listToJSON((List<?>) obj);
        }
        
        if (obj instanceof Object[]) {
            return listToJSON(Arrays.asList((Object[]) obj));
        }
        
        return "\"" + obj.toString() + "\"";
    }
    
    /**
     * Map转JSON对象
     */
    private static String mapToJSON(Map<?, ?> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!first) {
                sb.append(",");
            }
            first = false;
            
            sb.append("\"").append(entry.getKey()).append("\":");
            sb.append(toJSON(entry.getValue()));
        }
        
        sb.append("}");
        return sb.toString();
    }
    
    /**
     * List转JSON数组
     */
    private static String listToJSON(List<?> list) {
        StringBuilder sb = new StringBuilder("[");
        boolean first = true;
        
        for (Object item : list) {
            if (!first) {
                sb.append(",");
            }
            first = false;
            
            sb.append(toJSON(item));
        }
        
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * 转义JSON字符串
     */
    private static String escape(String str) {
        return str.replace("\\", "\\\\")
                  .replace("\"", "\\\"")
                  .replace("\n", "\\n")
                  .replace("\r", "\\r")
                  .replace("\t", "\\t");
    }
    
    /**
     * 解析JSON字符串为Map
     */
    public static Map<String, Object> parseJSON(String json) {
        json = json.trim();
        if (!json.startsWith("{")) {
            throw new IllegalArgumentException("Invalid JSON object");
        }
        
        return parseObject(json);
    }
    
    /**
     * 解析JSON对象
     */
    private static Map<String, Object> parseObject(String json) {
        Map<String, Object> result = new LinkedHashMap<>();
        json = json.trim();
        
        if (json.equals("{}")) {
            return result;
        }
        
        // 移除外层花括号
        json = json.substring(1, json.length() - 1).trim();
        
        int pos = 0;
        while (pos < json.length()) {
            // 跳过空白
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
            
            if (pos >= json.length()) break;
            
            // 解析key
            if (json.charAt(pos) != '"') {
                throw new IllegalArgumentException("Expected quote at position " + pos);
            }
            
            int keyStart = pos + 1;
            int keyEnd = json.indexOf('"', keyStart);
            String key = json.substring(keyStart, keyEnd);
            pos = keyEnd + 1;
            
            // 跳过冒号
            while (pos < json.length() && (Character.isWhitespace(json.charAt(pos)) || json.charAt(pos) == ':')) {
                pos++;
            }
            
            // 解析value
            ParseResult valueResult = parseValue(json, pos);
            result.put(key, valueResult.value);
            pos = valueResult.endPos;
            
            // 跳过逗号
            while (pos < json.length() && (Character.isWhitespace(json.charAt(pos)) || json.charAt(pos) == ',')) {
                pos++;
            }
        }
        
        return result;
    }
    
    /**
     * 解析JSON值
     */
    private static ParseResult parseValue(String json, int startPos) {
        int pos = startPos;
        char ch = json.charAt(pos);
        
        // 字符串
        if (ch == '"') {
            int end = json.indexOf('"', pos + 1);
            while (end > 0 && json.charAt(end - 1) == '\\') {
                end = json.indexOf('"', end + 1);
            }
            String value = json.substring(pos + 1, end);
            return new ParseResult(unescape(value), end + 1);
        }
        
        // 数组
        if (ch == '[') {
            return parseArray(json, pos);
        }
        
        // 对象
        if (ch == '{') {
            int depth = 0;
            int end = pos;
            do {
                if (json.charAt(end) == '{') depth++;
                if (json.charAt(end) == '}') depth--;
                end++;
            } while (depth > 0 && end < json.length());
            
            String objStr = json.substring(pos, end);
            return new ParseResult(parseObject(objStr), end);
        }
        
        // null
        if (json.startsWith("null", pos)) {
            return new ParseResult(null, pos + 4);
        }
        
        // true/false
        if (json.startsWith("true", pos)) {
            return new ParseResult(true, pos + 4);
        }
        if (json.startsWith("false", pos)) {
            return new ParseResult(false, pos + 5);
        }
        
        // 数字
        int end = pos;
        while (end < json.length() && 
               (Character.isDigit(json.charAt(end)) || 
                json.charAt(end) == '.' || 
                json.charAt(end) == '-' || 
                json.charAt(end) == 'e' || 
                json.charAt(end) == 'E')) {
            end++;
        }
        
        String numStr = json.substring(pos, end);
        Object value;
        if (numStr.contains(".") || numStr.contains("e") || numStr.contains("E")) {
            value = Double.parseDouble(numStr);
        } else {
            value = Integer.parseInt(numStr);
        }
        
        return new ParseResult(value, end);
    }
    
    /**
     * 解析JSON数组
     */
    private static ParseResult parseArray(String json, int startPos) {
        List<Object> result = new ArrayList<>();
        int pos = startPos + 1; // 跳过'['
        
        while (pos < json.length()) {
            // 跳过空白
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
            
            if (json.charAt(pos) == ']') {
                pos++;
                break;
            }
            
            ParseResult valueResult = parseValue(json, pos);
            result.add(valueResult.value);
            pos = valueResult.endPos;
            
            // 跳过逗号
            while (pos < json.length() && (Character.isWhitespace(json.charAt(pos)) || json.charAt(pos) == ',')) {
                pos++;
            }
        }
        
        return new ParseResult(result, pos);
    }
    
    /**
     * 反转义JSON字符串
     */
    private static String unescape(String str) {
        return str.replace("\\\"", "\"")
                  .replace("\\n", "\n")
                  .replace("\\r", "\r")
                  .replace("\\t", "\t")
                  .replace("\\\\", "\\");
    }
    
    /**
     * 解析结果
     */
    private static class ParseResult {
        Object value;
        int endPos;
        
        ParseResult(Object value, int endPos) {
            this.value = value;
            this.endPos = endPos;
        }
    }
}
